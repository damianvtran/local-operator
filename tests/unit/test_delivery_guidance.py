"""Keep always-loaded delivery policy small and route detail to one authority.

These are structural guards, not claims that a model obeys prose. Fresh-agent
pilots must still discriminate routing, failed preflight and approval behavior.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from local_operator.agent_profiles import (
    install_seed,
    load_seed,
    resolve_profile,
    sync_installed_seeds,
)
from local_operator.agents import AgentRegistry
from local_operator.skills.discovery import discover_skills, packaged_skill_root
from local_operator.teams import Team

ROOT = Path(__file__).resolve().parents[2]
# Budgets cover the installed body, not YAML metadata, and allow small edits
# without letting an incident archive grow into every delegated turn again.
BUDGETS = {
    "coder": 2200,
    "reviewer": 3300,
    "manager": 3300,
    "designer": 5300,
    "ux-reviewer": 4300,
    "aida": 8000,
    "qa-tester": 2400,
}


@pytest.mark.parametrize("name,budget", BUDGETS.items())
def test_loaded_role_routes_to_bounded_delivery_authority(name, budget):
    profile = load_seed(name)
    assert profile is not None
    body = profile.instructions
    raw = (ROOT / f"local_operator/agent_seeds/{name}.md").read_text()
    # A loader-clipped prompt must not pass a budget check on its clipped view.
    assert body == raw.split("---", 2)[2].strip()
    assert len(body) <= budget
    assert "guide://delivery" in body


def test_delivery_authority_keeps_cost_and_safety_controls_together():
    policy = (ROOT / "local_operator/guides/delivery/GUIDE.md").read_text()
    for required in (
        "CI runs the same suite",
        "negative control",
        "source-only",
        "stable session ID",
        "destructive",
        "BLOCKER",
        "byte-identical",
        "actual served model",
    ):
        assert required in policy
    assert len(policy) < 6500


def test_root_instructions_link_detail_instead_of_duplicating_it():
    assert "guide://delivery" in (ROOT / "AGENTS.md").read_text()


def test_qa_resolves_without_unknown_role_fallback_and_preserves_custom_copy(tmp_path):
    registry = AgentRegistry(tmp_path / "config")
    assert resolve_profile("nonexistent-delivery-role", registry=registry) is None
    qa = resolve_profile("qa-tester", registry=registry)
    assert qa is not None
    seed = load_seed("qa-tester")
    assert seed is not None and qa.instructions == seed.instructions
    assert qa.may_delegate is False
    assert qa.effort is None and qa.model == ""
    assert qa.tools is not None and "browser" in qa.tools
    assert not {"edit", "write", "task"}.intersection(qa.tools)
    assert install_seed("qa-tester", registry=registry) is not None
    row = registry.get_agent_by_name("qa-tester")
    assert row is not None
    custom = qa.instructions + "\nCUSTOM isolation constraint"
    registry.set_agent_system_prompt(row.id, custom)
    install_seed("qa-tester", registry=registry)
    sync_installed_seeds(registry)
    assert registry.get_agent_system_prompt(row.id) == custom
    loaded = resolve_profile("qa-tester", registry=registry)
    assert loaded is not None and loaded.instructions == custom


def test_custom_manager_team_boundary_carries_delivery_controls():
    team = Team(
        id="12345678-1234-1234-1234-123456789012",
        name="portable-delivery",
        manager="custom-coordinator",
        created_date=datetime.now(timezone.utc),
    )
    prompt = team.manager_preamble()
    assert "CI runs the same suite" in prompt
    assert "independent real-path QA" in prompt
    assert "stable session ID" in prompt
    assert "guide://delivery" in prompt
    assert len(prompt) < 2400


def test_packaged_skills_expose_compact_delivery_controls():
    skills, warnings = discover_skills([packaged_skill_root()])
    assert not warnings
    by_name = {skill.name: skill for skill in skills}
    for name, marker in (
        ("code-review", "genuine late BLOCKER/MAJOR"),
        ("responding-to-review", "provenance-only"),
        ("engineering-principles", "compiled-source identity"),
    ):
        skill = by_name[name]
        assert skill.file_path == packaged_skill_root() / name / "SKILL.md"
        body = skill.file_path.read_text()
        assert marker in body
        assert len(body) < 6500
