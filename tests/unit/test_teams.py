"""Team registry: roster, layered briefs, and name rules."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from local_operator.teams import (
    MAX_TEAM_INSTRUCTIONS_CHARS,
    Team,
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
    # A bare token stays an agent — the existing grammar is untouched.
    assert all(m.kind == "agent" for m in slots)


def test_parse_members_team_prefix() -> None:
    """``team:`` marks a nested-team slot; ``team:name:count`` carries a count."""
    slots = parse_members(["coder", "team:pod", "team:pod:2", "reviewer:2"])
    # ``team:pod`` and ``team:pod:2`` collapse into one team slot (summed).
    assert [(m.kind, m.role, m.count) for m in slots] == [
        ("agent", "coder", 1),
        ("team", "pod", 3),
        ("agent", "reviewer", 2),
    ]


def test_parse_members_team_prefix_is_case_insensitive() -> None:
    slots = parse_members(["TEAM:pod", "Team:qa:2"])
    assert [(m.kind, m.role, m.count) for m in slots] == [
        ("team", "pod", 1),
        ("team", "qa", 2),
    ]


def test_parse_members_agent_and_team_of_same_name_do_not_collapse() -> None:
    """A member ``pod`` and a sub-team ``pod`` are distinct slots, not merged."""
    slots = parse_members(["pod", "team:pod"])
    assert [(m.kind, m.role, m.count) for m in slots] == [
        ("agent", "pod", 1),
        ("team", "pod", 1),
    ]


def test_parse_members_team_prefix_without_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="no team name"):
        parse_members(["team:"])


def test_roster_lines_badges_team_slots() -> None:
    team = Team(
        id="x",
        name="org",
        created_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        manager="director",
        members=[
            TeamMember(role="coder"),
            TeamMember(role="pod", kind="team", count=2),
        ],
    )
    lines = team.roster_lines()
    assert lines[0] == "- manager: director (you, when this team is invoked)"
    assert "- coder" in lines[1]
    # A nested-team slot is badged so an org is distinguishable from a roster.
    assert lines[2] == "- pod (team) x2"


def test_fieldless_team_yml_loads_as_agent_kind(tmp_path: Path) -> None:
    """A pre-existing ``team.yml`` with no ``kind`` key loads to kind='agent'.

    This is the one backward-compat guarantee the model change rests on: an
    old file that predates nesting must validate unchanged, or ``_load``'s
    except-and-skip would silently drop the team. Written as a real on-disk
    file, not a model construction, because it is the DISK format that must
    survive the schema addition.
    """

    team_id = "legacy-id"
    team_dir = tmp_path / "teams" / team_id
    team_dir.mkdir(parents=True)
    (team_dir / "team.yml").write_text(
        "id: legacy-id\n"
        "name: legacy\n"
        "created_date: '2024-01-01T00:00:00+00:00'\n"
        "manager: manager\n"
        "members:\n"
        "- role: coder\n"
        "  count: 1\n"
        "- role: reviewer\n"
        "  count: 2\n",
        encoding="utf-8",
    )
    (team_dir / "instructions.md").write_text("", encoding="utf-8")
    (team_dir / "project.md").write_text("", encoding="utf-8")
    registry = TeamRegistry(tmp_path)
    loaded = registry.get_team_by_name("legacy")
    assert loaded is not None
    assert [m.role for m in loaded.members] == ["coder", "reviewer"]
    # The absent field defaults to "agent" — the roster means exactly what it
    # meant before nesting existed.
    assert all(m.kind == "agent" for m in loaded.members)


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


def test_list_is_metadata_only_and_get_loads_briefs_once(tmp_path: Path, monkeypatch) -> None:
    """Picker refreshes must not pay for either 8k brief until a full lookup."""
    first = TeamRegistry(tmp_path)
    team = first.create_team(
        TeamEditFields(
            name="ops",
            manager="manager",
            instructions="Review before merging.",
            project="on-call",
        )
    )

    brief_paths = {
        tmp_path / "teams" / team.id / "instructions.md",
        tmp_path / "teams" / team.id / "project.md",
    }
    real_read_text = Path.read_text
    reads: list[Path] = []

    def tracked_read_text(path: Path, *args, **kwargs) -> str:
        if path in brief_paths:
            reads.append(path)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    second = TeamRegistry(tmp_path)
    listed = second.list_teams()
    assert [row.name for row in listed] == ["ops"]
    assert listed[0].instructions == ""
    assert listed[0].project == ""
    assert reads == []

    loaded = second.get_team_by_name("OPS")
    assert loaded is not None
    assert loaded.instructions == "Review before merging."
    assert loaded.project == "on-call"
    assert reads == [
        tmp_path / "teams" / team.id / "instructions.md",
        tmp_path / "teams" / team.id / "project.md",
    ]

    assert second.get_team(team.id) is loaded
    assert len(reads) == 2
