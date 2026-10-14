"""Definition-revision digest and CAS guard used by local editor writes."""

from __future__ import annotations

import pytest

from local_operator.hub_sync.definition import (
    DefinitionRevisionMismatch,
    definition_revision,
    require_fresh_revision,
)

AGENT = {
    "name": "reviewer",
    "label": "",
    "kind": "role",
    "description": "Reviews diffs.",
    "instructions": "Read the diff.",
    "tools": ["read"],
    "effort": "hi",
    "delegate": False,
    "action_class": "reactive",
    # Runtime/provenance noise that must NOT affect the digest:
    "last_message": "hello",
    "last_message_datetime": "2030-01-01T00:00:00",
    "created_date": "2020-01-01T00:00:00",
    "seed_origin": "builtin",
}


def test_revision_is_stable_and_ignores_runtime_noise() -> None:
    base = definition_revision("agent", "a1", AGENT)
    noisy = {**AGENT, "last_message": "later", "created_date": "2099-01-01T00:00:00"}
    assert definition_revision("agent", "a1", noisy) == base


def test_revision_changes_on_authored_edit_and_stable_id() -> None:
    base = definition_revision("agent", "a1", AGENT)
    assert definition_revision("agent", "a1", {**AGENT, "description": "Different."}) != base
    assert definition_revision("agent", "a2", AGENT) != base


def test_team_and_agent_domains_do_not_collide() -> None:
    team = {"name": "squad", "members": [], "instructions": "x"}
    assert definition_revision("team", "squad", team).startswith("definition-revision:v1:")


def test_require_fresh_accepts_none_and_matching_token() -> None:
    token = definition_revision("agent", "a1", AGENT)
    require_fresh_revision("agent", "a1", AGENT, None)  # legacy unconditional
    require_fresh_revision("agent", "a1", AGENT, token)


def test_require_fresh_refuses_stale_and_malformed() -> None:
    stale = definition_revision("agent", "a1", {**AGENT, "description": "old"})
    with pytest.raises(DefinitionRevisionMismatch):
        require_fresh_revision("agent", "a1", AGENT, stale)
    with pytest.raises(DefinitionRevisionMismatch):
        require_fresh_revision("agent", "a1", AGENT, "not-a-token")
    assert DefinitionRevisionMismatch.code == "local-concurrent-edit"
