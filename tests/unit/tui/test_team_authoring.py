"""Pure team drafts exercise the same registry used by the tool and desktop."""

from dataclasses import replace

import pytest

from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tui.team_authoring import TeamDraft, build_team_plan, member_text


def test_create_real_registry(tmp_path):
    registry = TeamRegistry(tmp_path)
    draft = TeamDraft(name="crew", instructions="Coordinate.", members="coder:2, team:pod")
    plan = build_team_plan(None, draft)
    assert not plan.error_code
    row = registry.create_team(TeamEditFields(**plan.changes))
    assert row.instructions == "Coordinate."
    assert [(m.kind, m.role, m.count) for m in row.members] == [
        ("agent", "coder", 2),
        ("team", "pod", 1),
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("label", "My crew"),
        ("description", "For delivery"),
        ("manager", "architect"),
        ("members", "coder:2, team:pod"),
        ("aliases", "delivery, builders"),
        ("instructions", "Ship safely."),
        ("project", "Build a tool."),
    ],
)
def test_changed_only(field, value):
    initial = TeamDraft(name="crew", instructions="Help.")
    plan = build_team_plan(initial, replace(initial, **{field: value}))
    assert not plan.error_code
    assert set(plan.changes) == {field}


def test_noop_omits_roster_and_aliases():
    initial = TeamDraft(name="crew", instructions="Help.", members="coder", aliases="builders")
    assert build_team_plan(initial, initial).changes == {}
    assert build_team_plan(initial, replace(initial, description="New")).changes == {
        "description": "New"
    }


@pytest.mark.parametrize(
    "field", ["label", "description", "instructions", "project", "aliases", "members"]
)
def test_explicit_clears_are_supported_by_core(tmp_path, field):
    registry = TeamRegistry(tmp_path)
    row = registry.create_team(
        TeamEditFields(
            name="crew",
            label="Crew label",
            description="Use",
            instructions="Help.",
            project="Project",
            aliases=["builders"],
            members=[TeamMember(role="coder")],
        )
    )
    initial = TeamDraft(
        name="crew",
        label=row.label,
        description=row.description,
        instructions=row.instructions,
        project=row.project,
        aliases="builders",
        members="coder",
    )
    plan = build_team_plan(initial, replace(initial, **{field: ""}), row.members)
    assert not plan.error_code
    row = registry.update_team(row.id, TeamEditFields(**plan.changes))
    value = getattr(row, field)
    assert value == ("Crew" if field == "label" else [] if field in ("aliases", "members") else "")


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("name", "", "invalid_name"),
        ("name", "two words", "invalid_name"),
        ("manager", " ", "required_manager"),
        ("instructions", " ", "required_instructions"),
        ("members", "coder,,reviewer", "invalid_members"),
        ("members", "coder:no", "invalid_members"),
        ("members", "coder:0", "invalid_members"),
        ("members", "coder:17", "invalid_members"),
        ("members", "team:", "invalid_members"),
        ("aliases", "one,,two", "invalid_aliases"),
        ("aliases", "two words", "invalid_aliases"),
    ],
)
def test_invalid_draft(field, value, code):
    draft = replace(TeamDraft(name="crew", instructions="Help."), **{field: value})
    plan = build_team_plan(None, draft)
    assert (plan.error_field, plan.error_code) == (field, code)


def test_override_fixture_before_after_real_registry(tmp_path):
    registry = TeamRegistry(tmp_path)
    row = registry.create_team(
        TeamEditFields(
            name="crew",
            instructions="Help.",
            members=[
                TeamMember(role="coder", count=2, tier="hi", effort="high"),
                TeamMember(kind="team", role="pod", tier="med", effort="low"),
                TeamMember(role="reviewer", tier="lo", effort="auto"),
            ],
        )
    )
    original = [m.model_dump(mode="json") for m in row.members]
    original_bytes = [m.model_dump_json().encode() for m in row.members]
    initial = TeamDraft(
        name=row.name,
        label=row.label,
        instructions=row.instructions,
        members=member_text(row.members),
    )
    plan = build_team_plan(initial, replace(initial, description="Only description"), row.members)
    assert plan.changes == {"description": "Only description"}
    row = registry.update_team(row.id, TeamEditFields(**plan.changes))
    registry = TeamRegistry(tmp_path)
    row = registry.get_team(row.id)
    assert [m.model_dump(mode="json") for m in row.members] == original
    assert [m.model_dump_json().encode() for m in row.members] == original_bytes
    plan = build_team_plan(
        initial, replace(initial, members="coder:3, team:pod:2, designer"), row.members
    )
    row = registry.update_team(row.id, TeamEditFields(**plan.changes))
    row = TeamRegistry(tmp_path).get_team(row.id)
    assert [m.model_dump(mode="json") for m in row.members] == [
        {**original[0], "count": 3},
        {**original[1], "count": 2},
        TeamMember(role="designer").model_dump(mode="json"),
    ]


def test_roster_count_collapse():
    plan = build_team_plan(
        None, TeamDraft(name="crew", instructions="Help.", members="coder, coder:2, team:pod:2")
    )
    assert member_text(plan.changes["members"]) == "coder:3, team:pod:2"


@pytest.mark.parametrize(
    "fields", [TeamEditFields(name="crew"), TeamEditFields(name="other", aliases=["builders"])]
)
def test_registry_owns_duplicate_validation(tmp_path, fields):
    registry = TeamRegistry(tmp_path)
    registry.create_team(TeamEditFields(name="crew", aliases=["builders"]))
    with pytest.raises(ValueError):
        registry.create_team(fields)
