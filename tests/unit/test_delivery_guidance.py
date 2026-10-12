"""Keep always-loaded delivery policy small and route detail to one authority.

These are structural guards, not claims that a model obeys prose. Fresh-agent
pilots must still discriminate routing, failed preflight and approval behavior.
"""

from pathlib import Path

import pytest

from local_operator.agent_profiles import load_seed

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
}


@pytest.mark.parametrize("name,budget", BUDGETS.items())
def test_loaded_role_routes_to_bounded_delivery_authority(name, budget):
    body = load_seed(name).instructions
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
