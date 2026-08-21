"""Team registry: roster, layered briefs, and name rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.teams import (
    MAX_TEAM_INSTRUCTIONS_CHARS,
    TeamEditFields,
    TeamMember,
    TeamRegistry,
    parse_members,
)


@pytest.fixture()
def registry(tmp_path: Path) -> TeamRegistry:
    return TeamRegistry(tmp_path)


def test_create_list_show_update_delete(registry: TeamRegistry) -> None:
    team = registry.create_team(
        TeamEditFields(
            name="feature-release",
            description="Ship a user-facing change",
            manager="manager",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer", count=2)],
            instructions="Architect designs; coder implements; reviewer signs off.",
            project="user-dashboard",
        )
    )
    assert team.name == "feature-release"
    assert team.manager == "manager"
    assert [m.role for m in team.members] == ["coder", "reviewer"]
    assert team.members[1].count == 2
    assert "user-dashboard" in team.project

    listed = registry.list_teams()
    assert [row.name for row in listed] == ["feature-release"]
    assert registry.get_team_by_name("Feature-Release") is team or registry.get_team_by_name(
        "feature-release"
    )

    updated = registry.update_team(team.id, TeamEditFields(project="admin-api", instructions=None))
    assert updated.project == "admin-api"
    assert "Architect designs" in updated.instructions

    registry.delete_team(team.id)
    assert registry.list_teams() == []


def test_duplicate_name_is_refused(registry: TeamRegistry) -> None:
    registry.create_team(TeamEditFields(name="alpha", manager="manager"))
    with pytest.raises(ValueError, match="already exists"):
        registry.create_team(TeamEditFields(name="alpha", manager="manager"))


def test_name_rejects_spaces(registry: TeamRegistry) -> None:
    with pytest.raises(ValueError, match="team name"):
        registry.create_team(TeamEditFields(name="Feature Release", manager="manager"))


def test_parse_members_collapses_counts() -> None:
    slots = parse_members(["coder", "reviewer:2", "coder"])
    assert [(m.role, m.count) for m in slots] == [("coder", 2), ("reviewer", 2)]


def test_manager_preamble_layers_briefs(registry: TeamRegistry) -> None:
    team = registry.create_team(
        TeamEditFields(
            name="release",
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="Do not merge without review.",
            project="Keep the dashboard shipping weekly.",
        )
    )
    brief = team.manager_preamble()
    assert "[team: release]" in brief
    assert "coder" in brief
    assert "Do not merge without review." in brief
    assert "Keep the dashboard shipping weekly." in brief
    member = team.member_preamble("coder")
    assert "You are coder on this team" in member
    assert "Do not merge without review." in member


def test_oversized_briefs_are_refused(registry: TeamRegistry) -> None:
    with pytest.raises(ValueError, match="exceed"):
        registry.create_team(
            TeamEditFields(
                name="huge",
                manager="manager",
                instructions="x" * (MAX_TEAM_INSTRUCTIONS_CHARS + 1),
            )
        )


def test_reload_from_disk(tmp_path: Path) -> None:
    first = TeamRegistry(tmp_path)
    first.create_team(
        TeamEditFields(
            name="ops",
            manager="manager",
            members=[TeamMember(role="scout")],
            project="on-call",
        )
    )
    second = TeamRegistry(tmp_path)
    loaded = second.get_team_by_name("ops")
    assert loaded is not None
    assert loaded.project == "on-call"
    assert loaded.members[0].role == "scout"
