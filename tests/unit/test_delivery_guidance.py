"""Keep always-loaded delivery policy small and route detail to one authority.

These are structural guards, not claims that a model obeys prose. Fresh-agent
pilots must still discriminate routing, failed preflight and approval behavior.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from local_operator.agent_profiles import (
    DELIVERY_POINTER,
    install_seed,
    load_seed,
    resolve_profile,
    sync_installed_seeds,
)
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.skills.discovery import discover_skills, packaged_skill_root
from local_operator.teams import Team

ROOT = Path(__file__).resolve().parents[2]


def _fields(**overrides):
    # AgentEditFields validates strictly, so every field is spelled out.
    names = AgentEditFields.model_fields
    return AgentEditFields(**{name: overrides.get(name) for name in names})


# Budgets cover the installed body, not YAML metadata, and allow small edits
# without letting an incident archive grow into every delegated turn again.
# The review-economy pointers (context contract, scope, freshness, round
# hygiene) raised coder/manager/designer/ux/qa slightly after trimming the
# duplicated coverage prose; reviewer stayed inside its 3500 by trimming.
BUDGETS = {
    "coder": 2400,
    "reviewer": 3500,
    "manager": 3450,
    "designer": 5650,
    "ux-reviewer": 4650,
    "aida": 8000,
    "qa-tester": 2700,
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
        "Review economy",
        "Context contract",
        "pre-existing",
        "merge of main",
    ):
        assert required in policy
    # Raised from 6800 for the review-economy section: measured review churn
    # (median 6-7 rounds/PR, ~55% no-BLOCKER/MAJOR re-runs after a clean round) costs far more
    # tokens than ~1.7k always-loaded chars. Still a hard cap: detail belongs
    # in AGENTS.md, not here.
    assert len(policy) < 8500


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
    assert install_seed("qa-tester", registry=registry, scope="global") is not None
    row = registry.get_agent_by_name("qa-tester")
    assert row is not None
    custom = qa.instructions + "\nCUSTOM isolation constraint"
    registry.set_agent_system_prompt(row.id, custom)
    install_seed("qa-tester", registry=registry, scope="global")
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


def _child_prompt(tmp_path, registry, role):
    # The real launch assembler (what ``task(agent=role)`` sends the child), so
    # the check sits at the registered-role boundary, not on seed Markdown.
    from local_operator.harness.subagent import _effective_prompt
    from tests.unit.session.test_session import ScriptedStream, make_session

    parent = make_session(tmp_path, ScriptedStream([]), agent_registry=registry)
    return _effective_prompt("Synthetic bounded task", role, parent)


def test_registered_engineering_role_predating_seed_gets_delivery_pointer(tmp_path):
    """Pilots that loaded seeds passed while operator rows from October kept
    running their old prompts; this launches a hand-edited registered row and
    checks the pointer arrives without losing anything, and without writing."""

    registry = AgentRegistry(tmp_path / "config")
    legacy = "Old October review rules.\nCUSTOM safety block: never touch prod."
    row = registry.create_agent(
        _fields(
            name="reviewer",
            description="custom reviewer",
            tags=["role", "tools:read,grep", "effort:hi"],
            model="anthropic/claude-opus-5-5",
        )
    )
    registry.set_agent_system_prompt(row.id, legacy)

    prompt, profile = _child_prompt(tmp_path, registry, "reviewer")
    assert profile is not None
    assert legacy in prompt and prompt.count(DELIVERY_POINTER) == 1
    assert profile.instructions == f"{legacy}\n\n{DELIVERY_POINTER}"
    assert profile.tools == ("read", "grep") and profile.effort == "hi"
    assert profile.model == "anthropic/claude-opus-5-5"
    assert profile.description == "custom reviewer"
    # Show/edit surfaces read resolve_profile: they keep the exact stored text.
    shown = resolve_profile("reviewer", registry=registry)
    assert shown is not None and shown.instructions == legacy
    assert registry.get_agent_system_prompt(row.id) == legacy

    # A prompt already carrying the pointer is not given a second one.
    registry.set_agent_system_prompt(row.id, profile.instructions)
    again, _ = _child_prompt(tmp_path, registry, "reviewer")
    assert again.count(DELIVERY_POINTER) == 1


def test_delivery_pointer_skips_current_seed_rows_and_other_roles(tmp_path):
    registry = AgentRegistry(tmp_path / "config")
    seed = load_seed("coder")
    assert seed is not None
    # Packaged seed (no row) and an installed, edited-by-addition seed row both
    # already carry the current rules: no pointer, bytes unchanged.
    fresh, _ = _child_prompt(tmp_path, registry, "coder")
    assert DELIVERY_POINTER not in fresh
    installed = install_seed("coder", registry=registry, scope="global")
    assert installed is not None and installed[0].agent_id is not None
    extended = seed.instructions + "\nCUSTOM house rule"
    registry.set_agent_system_prompt(installed[0].agent_id, extended)
    _, coder = _child_prompt(tmp_path, registry, "coder")
    assert coder is not None and coder.instructions == extended

    # A non-engineering registered role keeps its own text exactly.
    row = registry.create_agent(_fields(name="scout-custom", tags=["role"]))
    registry.set_agent_system_prompt(row.id, "Own rules.")
    _, custom = _child_prompt(tmp_path, registry, "scout-custom")
    assert custom is not None and custom.instructions == "Own rules."
