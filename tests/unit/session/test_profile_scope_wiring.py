"""Where the session scope is wired (LOH-9 S1)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.profile_scope import (
    ScopedAgentRegistry,
    ScopedTeamRegistry,
    scope_session_registries,
)
from local_operator.session.attached import AttachedSession
from local_operator.teams import TeamRegistry
from local_operator.tools.team_tool import _registry as team_tool_registry


def test_the_viewer_reads_the_session_scope(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "aaaaaaaaaaaa"
    session_agents = AgentRegistry(session_dir / "roles", initialize=False, migrate=False)
    session_agents.create_agent(AgentEditFields(name="viewer-only", tags=["role"]))

    viewer = object.__new__(AttachedSession)
    viewer._config_dir = tmp_path
    viewer._session_id = "aaaaaaaaaaaa"
    viewer._agent_registry_cache = None
    viewer._agent_registry_failed_at = None
    viewer._team_registry_cache = None
    viewer._team_registry_failed_at = None

    agents = viewer.agent_registry
    teams = viewer.team_registry
    assert isinstance(agents, ScopedAgentRegistry)
    assert isinstance(teams, ScopedTeamRegistry)
    assert agents.get_agent_by_name("viewer-only") is not None
    assert AgentRegistry(tmp_path).get_agent_by_name("viewer-only") is None


def test_the_team_tool_accepts_a_composite(tmp_path: Path) -> None:
    agents, teams = scope_session_registries(
        AgentRegistry(tmp_path), TeamRegistry(tmp_path), tmp_path / "sessions" / "aaaaaaaaaaaa"
    )
    assert team_tool_registry(SimpleNamespace(team_registry=teams)) is teams
    assert team_tool_registry(SimpleNamespace(team_registry=None)) is None
    assert team_tool_registry(SimpleNamespace(team_registry=object())) is None
