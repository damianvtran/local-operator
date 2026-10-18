"""Profile draft rules are independent of Textual and persistence."""

from dataclasses import replace

import pytest

from local_operator.tui.agent_authoring import AgentDraft, build_agent_plan, parse_tools


def test_create_role():
    plan = build_agent_plan(None, AgentDraft(name="author", instructions="Be clear."), ())
    assert not plan.error_code
    assert plan.changes == dict(
        name="author",
        kind="role",
        label="",
        description="",
        instructions="Be clear.",
        tools=[],
        effort="inherit",
        delegate=False,
        action_class="reactive",
    )


def test_create_specialist():
    plan = build_agent_plan(
        None, AgentDraft(name="author", kind="specialist", instructions="Help."), ()
    )
    assert not plan.error_code
    assert not ({"tools", "effort", "delegate"} & plan.changes.keys())


@pytest.mark.parametrize(
    "field,value",
    [
        ("label", "Author"),
        ("description", "For writing"),
        ("instructions", "Write."),
        ("tools", "read, bash"),
        ("effort", "hi"),
        ("delegate", True),
        ("action_class", "proactive"),
    ],
)
def test_edit_changed_only(field, value):
    initial = AgentDraft(name="author", instructions="Help.")
    plan = build_agent_plan(initial, replace(initial, **{field: value}), ("hi",))
    assert not plan.error_code
    assert set(plan.changes) == {field}


def test_noop():
    initial = AgentDraft(name="author", instructions="Help.")
    assert build_agent_plan(initial, initial, ()).changes == {}


@pytest.mark.parametrize("text", ["a,,b", ",a", "a,", "two words"])
def test_invalid_tools(text):
    with pytest.raises(ValueError):
        parse_tools(text)


def test_tools():
    assert parse_tools(" read, bash,read ") == ["read", "bash"]
    assert parse_tools(" ") == []


@pytest.mark.parametrize("change", [{"tools": "bash"}, {"effort": "hi"}, {"delegate": True}])
def test_specialist_refuses_role_fields(change):
    draft = AgentDraft(name="author", kind="specialist", instructions="Help.", **change)
    assert build_agent_plan(None, draft, ("hi",)).error_code == "specialist_fields"


def test_invalid_effort():
    assert (
        build_agent_plan(
            None, AgentDraft(name="author", instructions="Help.", effort="absent"), ()
        ).error_code
        == "invalid_effort"
    )


@pytest.mark.parametrize("name", ["", "two words", "a/b", "a\\b"])
def test_invalid_name(name):
    assert (
        build_agent_plan(None, AgentDraft(name=name, instructions="Help."), ()).error_field
        == "name"
    )


def test_instructions_required():
    assert build_agent_plan(None, AgentDraft(name="author"), ()).error_field == "instructions"


def test_edit_clearing_description_refuses_without_losing_draft():
    initial = AgentDraft(name="author", instructions="Help.", description="For writing")
    draft = replace(initial, description="")
    plan = build_agent_plan(initial, draft, ())
    assert plan.error_code == "description_clear_unsupported"
    assert plan.error_field == "description"
    assert draft.description == ""


def test_edit_clearing_instructions_refuses():
    initial = AgentDraft(name="author", instructions="Help.")
    assert (
        build_agent_plan(initial, replace(initial, instructions=""), ()).error_field
        == "instructions"
    )
