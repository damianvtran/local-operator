"""Session-scoped agent/team registries (LOH-9 S1).

The composite is the seam that lets a conversation see definitions authored for
it without making them global. These tests pin the READ contract and the S1
merge-safety rule: writes still land in the global store, so landing S1 alone
changes no behaviour. Writes are routed by row ownership only, which is what
makes the fixtures' session rows editable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.profile_scope import (
    ScopedAgentRegistry,
    ScopedTeamRegistry,
    scope_agent_registry,
    scope_session_registries,
    scope_team_registry,
)
from local_operator.teams import TeamEditFields, TeamRegistry, parse_members


def _fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with every field spelled out.

    The model validates in strict mode, so the analyzer treats the fields as
    required even though each defaults to ``None`` at runtime. Spelling them
    once here keeps each call site down to the fields the test cares about.
    """
    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


def _role(registry: AgentRegistry, name: str) -> str:
    return registry.create_agent(
        _fields(name=name, description=f"{name} for this task", tags=["role"])
    ).id


def _team(registry: TeamRegistry, name: str) -> str:
    return registry.create_team(
        TeamEditFields(
            name=name,
            description=f"{name} roster",
            manager="manager",
            members=parse_members(["reviewer"]),
            instructions="review in order",
            project="",
        )
    ).id


@pytest.fixture
def stores(tmp_path: Path) -> tuple[AgentRegistry, TeamRegistry, Path]:
    config = tmp_path / "config"
    config.mkdir()
    session_dir = config / "sessions" / "aaaaaaaaaaaa"
    return AgentRegistry(config), TeamRegistry(config), session_dir


@pytest.fixture
def scoped(
    stores: tuple[AgentRegistry, TeamRegistry, Path],
) -> tuple[ScopedAgentRegistry, ScopedTeamRegistry, AgentRegistry, TeamRegistry, Path]:
    global_agents, global_teams, session_dir = stores
    agents, teams = scope_session_registries(global_agents, global_teams, session_dir)
    assert isinstance(agents, ScopedAgentRegistry)
    assert isinstance(teams, ScopedTeamRegistry)
    return agents, teams, global_agents, global_teams, session_dir


# --- read overlay ---------------------------------------------------------------


def test_a_session_row_shadows_a_global_name(scoped) -> None:
    agents, _, global_agents, _, _ = scoped
    global_id = _role(global_agents, "reviewer")
    session_id = _role(agents.session_registry, "reviewer")

    assert agents.get_agent_by_name("reviewer").id == session_id
    assert agents.get_agent(session_id).id == session_id
    assert agents.get_agent(global_id).id == global_id
    assert agents.scope_of(agents.get_agent(session_id)) == "session"
    assert agents.scope_of(agents.get_agent(global_id)) == "global"


def test_listing_carries_both_stores_session_first(scoped) -> None:
    agents, _, global_agents, _, _ = scoped
    global_id = _role(global_agents, "global-role")
    session_id = _role(agents.session_registry, "session-role")

    ids = [row.id for row in agents.list_agents()]
    assert ids == [session_id, global_id]


def test_the_global_store_never_sees_a_session_row(scoped) -> None:
    agents, _, global_agents, _, _ = scoped
    _role(agents.session_registry, "session-only")

    assert global_agents.get_agent_by_name("session-only") is None
    assert [row.name for row in AgentRegistry(global_agents.config_dir).list_agents()] == []


def test_id_reads_dispatch_to_the_owning_store(scoped) -> None:
    agents, _, global_agents, _, _ = scoped
    global_id = _role(global_agents, "global-role")
    session_id = _role(agents.session_registry, "session-role")
    agents.session_registry.set_agent_system_prompt(session_id, "session prompt")
    global_agents.set_agent_system_prompt(global_id, "global prompt")

    assert agents.get_agent_system_prompt(session_id) == "session prompt"
    assert agents.get_agent_system_prompt(global_id) == "global prompt"
    assert agents.definition_revision_fields(session_id)["name"] == "session-role"
    assert agents.definition_revision_fields(global_id)["name"] == "global-role"


def test_agent_dir_points_at_the_owning_store(scoped) -> None:
    agents, _, global_agents, _, session_dir = scoped
    global_id = _role(global_agents, "global-role")
    session_id = _role(agents.session_registry, "session-role")

    assert agents.agent_dir(session_id) == session_dir / "roles" / "agents" / session_id
    assert agents.agent_dir(global_id) == global_agents.config_dir / "agents" / global_id


def test_teams_overlay_and_shadow_by_name(scoped) -> None:
    _, teams, _, global_teams, _ = scoped
    global_id = _team(global_teams, "pod")
    session_id = _team(teams.session_registry, "pod")

    assert teams.get_team_by_name("pod").id == session_id
    assert teams.get_team(global_id).id == global_id
    assert teams.get_team(session_id).id == session_id
    assert teams.scope_of(teams.get_team(session_id)) == "session"
    assert [row.id for row in teams.list_teams()] == [session_id, global_id]


# --- S1 merge-safety: writes stay global ----------------------------------------


def test_create_is_global_and_leaves_no_session_tree(
    stores: tuple[AgentRegistry, TeamRegistry, Path],
) -> None:
    global_agents, global_teams, session_dir = stores
    agents, teams = scope_session_registries(global_agents, global_teams, session_dir)

    agents.create_agent(_fields(name="one-off", tags=["role"]))
    teams.create_team(TeamEditFields(name="one-off-team", manager="manager", members=[]))

    assert global_agents.get_agent_by_name("one-off") is not None
    assert global_teams.get_team_by_name("one-off-team") is not None
    assert not session_dir.exists()


def test_building_the_scope_creates_nothing(
    stores: tuple[AgentRegistry, TeamRegistry, Path],
) -> None:
    global_agents, global_teams, session_dir = stores
    agents = scope_agent_registry(global_agents, session_dir)
    teams = scope_team_registry(global_teams, session_dir)

    assert agents.list_agents() == []
    assert teams.list_teams() == []
    assert not session_dir.exists()


def test_update_and_delete_route_by_ownership(scoped) -> None:
    agents, _, global_agents, _, _ = scoped
    global_id = _role(global_agents, "shared-name")
    session_id = _role(agents.session_registry, "shared-name")

    agents.update_agent(session_id, _fields(description="session edited"))
    assert agents.get_agent(session_id).description == "session edited"
    assert agents.get_agent(global_id).description == "shared-name for this task"

    agents.delete_agent(session_id)
    assert agents.get_agent_by_name("shared-name").id == global_id
    assert global_agents.get_agent(global_id) is not None


# --- strictness and recovery ----------------------------------------------------


def test_require_complete_metadata_sees_a_broken_session_row(
    stores: tuple[AgentRegistry, TeamRegistry, Path],
) -> None:
    from local_operator.session.errors import ProfileRegistryUnavailable

    global_agents, global_teams, session_dir = stores
    agents = scope_agent_registry(global_agents, session_dir)
    _role(agents.session_registry, "broken")
    row_dir = next((session_dir / "roles" / "agents").iterdir())
    (row_dir / "agent.yml").write_text(": not yaml :\n", encoding="utf-8")

    with pytest.raises(ProfileRegistryUnavailable):
        scope_agent_registry(global_agents, session_dir).require_complete_metadata()


def test_a_scoped_registry_is_not_nested_twice(
    stores: tuple[AgentRegistry, TeamRegistry, Path],
) -> None:
    global_agents, global_teams, session_dir = stores
    agents, teams = scope_session_registries(global_agents, global_teams, session_dir)

    again_agents, again_teams = scope_session_registries(
        agents, teams, session_dir / "elsewhere"
    )
    assert again_agents is agents
    assert again_teams is teams


# --- picker/exclusion projections ----------------------------------------------


def test_session_rows_reach_the_session_pickers_and_no_global_catalogue(scoped) -> None:
    """The two projections every picker reads, per scope.

    ``agent_listing_rows`` feeds the TUI's ``/agent`` list, the settings pane and
    the runtime's routed ``/agent``; ``profile_catalogue``/``team_catalogue``
    feed the desktop's global sidebar and the TUI's new-chat picker. A session
    row must appear in the first and in NEITHER of the second.
    """
    from local_operator.agent_profiles import agent_listing_rows
    from local_operator.server.utils.desktop_profiles import (
        profile_catalogue,
        team_catalogue,
    )

    agents, teams, global_agents, global_teams, _ = scoped
    _role(agents.session_registry, "conversation-role")
    _role(global_agents, "account-role")
    _team(teams.session_registry, "conversation-team")
    _team(global_teams, "account-team")

    listed = {row[0] for row in agent_listing_rows(agents)}
    assert {"conversation-role", "account-role"} <= listed

    profiles = {row["name"] for row in profile_catalogue(agents)}
    assert {"conversation-role", "account-role"} <= profiles

    global_profiles = {row["name"] for row in profile_catalogue(global_agents)}
    assert "account-role" in global_profiles
    assert "conversation-role" not in global_profiles

    assert {row["name"] for row in team_catalogue(teams)} == {
        "conversation-team",
        "account-team",
    }
    assert {row["name"] for row in team_catalogue(global_teams)} == {"account-team"}


def test_a_scoped_registry_write_never_creates_the_session_tree(
    stores: tuple[AgentRegistry, TeamRegistry, Path],
) -> None:
    """S1 merge-safety: the composite is read-only over the scope.

    The only way a session row exists in S1 is a test/fixture writing the store
    directly. Every write through the composite still lands in the global tree
    and leaves ``<session>/roles`` absent.
    """
    global_agents, global_teams, session_dir = stores
    agents, teams = scope_session_registries(global_agents, global_teams, session_dir)

    new_role = agents.create_agent(_fields(name="through-composite", tags=["role"]))
    agents.update_agent(new_role.id, _fields(description="edited"))
    new_team = teams.create_team(
        TeamEditFields(name="through-composite-team", manager="manager", members=[])
    )

    assert global_agents.get_agent(new_role.id).description == "edited"
    assert global_teams.get_team(new_team.id) is not None
    assert not session_dir.exists()
