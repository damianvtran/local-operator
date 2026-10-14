"""The thin per-kind adapter protocol over the one engine: identity + snapshot parity."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.adapters import adapter_for
from local_operator.hub_sync.identity import LinkError
from local_operator.teams import TeamEditFields, TeamRegistry

BASE = "## Rules\nBe brief.\n\n## Tools\nUse tools carefully."


def test_agent_adapter_snapshot_matches_its_canonical_fields(tmp_path: Path) -> None:
    registry = AgentRegistry(tmp_path)
    # The FULL field set, like the sibling direct constructions
    # (tests/unit/test_session_factory.py): a partial spelling trips pyright's
    # synthesized pydantic init for every ``= Field(None, ...)`` field.
    row = registry.create_agent(
        AgentEditFields(
            name="coder",
            security_prompt="",
            hosting="test",
            model="test",
            description="d",
            last_message="",
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
            tags=[],
            categories=[],
        )
    )
    registry.set_agent_system_prompt(row.id, BASE)
    adapter = adapter_for("agent")
    snapshot = adapter.snapshot(tmp_path, registry.get_agent(row.id))
    assert snapshot.local_id == row.id
    assert snapshot.fingerprint == prov.fingerprint_fields(
        "agent", adapter.canonical_fields(snapshot.fields)
    )


def test_team_adapter_snapshot_matches_its_canonical_fields(tmp_path: Path) -> None:
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="squad", manager="manager"))
    adapter = adapter_for("team")
    snapshot = adapter.snapshot(tmp_path, registry.get_team(team.id))
    assert snapshot.local_id == team.id
    assert snapshot.fingerprint == prov.fingerprint_fields(
        "team", adapter.canonical_fields(snapshot.fields)
    )


def test_unknown_kind_and_skill_local_id_are_refused(tmp_path: Path) -> None:
    with pytest.raises(LinkError):
        adapter_for("nope")
    # A skill's id is persisted, never derived from a row object here.
    with pytest.raises(LinkError):
        adapter_for("skill").local_id(object())
