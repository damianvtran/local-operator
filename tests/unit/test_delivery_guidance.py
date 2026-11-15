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
# The targeted-test-evidence rule (PR CI runs only fast guards, so the PR's
# `## Targeted test evidence` block is the test gate) replaced the shorter
# "whole-suite runs are CI's" lines and raised coder/reviewer/manager by the
# measured ~165/120/115 chars: each role must state its half of that gate.
# The coder's `WIDE:` -> full-suite.yml dispatch clause then measured 2616.
# The PR CI wording also changed to "runs only fast guards"; the old "runs no
# unit tests" was false once the guards job ran ~984 guard tests per code PR.
# The merge gate (2026-11-08) replaced "never ask CI for the full suite" in
# coder/manager/reviewer/qa-tester with the rule that actually holds `main`
# green: the suite runs ONCE at a labeled-PR merge gate, a red gate blocks the
# merge, and a push re-arms it. Measured with `len(body.split('---',2)[2])`:
# coder 2782, manager 3726, reviewer 3655, qa-tester 2694 — so coder +132,
# manager +126 and reviewer +5 of headroom, after compressing the lease
# paragraph (the coder's pinned `MINOR/NIT never force another round` sentence
# stays: a test asserts it). Trimming here would mean dropping a merge
# invariant to keep a budget, which is the wrong trade.
# Worktree hygiene (2026-11-13) gave every reader seed the site and removal
# duty for an exception worktree (+80 chars each, measured the same way:
# reviewer 3655 -> 3735, qa-tester 2694 -> 2774); readers had left checkouts
# behind and 33 in session pads because no reader surface said where or when.
BUDGETS = {
    "coder": 2800,
    "reviewer": 3740,
    "manager": 3750,
    "designer": 5650,
    "ux-reviewer": 4650,
    "aida": 8000,
    "qa-tester": 2780,
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
        "make test-affected",
        "Targeted test evidence",
        "ci-main-red",
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
    #
    # Raised again to 12200 for the review-round economy rules (+3,179 chars
    # measured with `git show HEAD:...GUIDE.md | wc -c` before/after): evidence
    # budget, measure-before-cap, harness-only delta, split trigger and
    # converge-once. That is what local-operator-mobile PR #156 cost by their
    # absence — 40 rounds for a product clean at round 3, 20 failed native
    # attempts (41.4 h) after the last product commit — against ~800 tokens of
    # on-demand guide read. Trim prose, not the invariants, before raising it: a
    # rule that names no measurement is the defect this section exists to stop.
    #
    # Raised to 12300 for the `WIDE:` clause (+81 chars net after trimming the
    # lease and CI paragraphs; 12199 -> 12280 measured with `len(read_text())`):
    # without it "no local full suite" reads as forbidding the one sanctioned
    # full-suite dispatch that `make test-affected` asks for on infra changes.
    #
    # Raised to 12600 for the merge gate (+260 measured with `len(read_text())`,
    # 12280 -> 12540): "no local full suite" had become the whole rule, and the
    # rule that actually holds the main branch green is that the suite runs once
    # at a labeled-PR merge gate, a red gate blocks the merge, and a push
    # re-arms it with `make merge-gate PR=<n>`. Trim prose, not that invariant.
    #
    # Raised to 12700 for the reader `evidence` sentence (+88 measured with
    # `len(read_text())`, 12559 -> 12647): without it a reviewer has no
    # sanctioned way to file the receipt its repository mandates, and refuses.
    #
    # Raised to 12900 for the subagent-lanes pointer (+148 measured with
    # `len(read_text())`, 12647 -> 12795): mesh-node managers, refused a pinned
    # tier, ran review/QA as parallel sessions or on `model=inherit`. The rule
    # itself lives in guide://sessions; this policy, which every engineering role
    # loads, carries only the one line that sends a reader there.
    assert len(policy) < 12900
    flat = " ".join(policy.split())
    assert "review/QA/design/UX are `task` subagents, on mesh nodes too" in flat
    assert "`guide://sessions`" in flat


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
    assert "Targeted test evidence" in prompt
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


@pytest.mark.parametrize(
    "name", ["manager", "coder", "reviewer", "qa-tester", "designer", "ux-reviewer", "architect"]
)
def test_engineering_seeds_carry_batching_and_minor_rule(name):
    # Every built-in engineering role states the one rule that most often
    # caused extra waves: accepted minors never earn their own round.
    profile = load_seed(name)
    assert profile is not None
    assert "guide://delivery" in profile.instructions
    assert "MINOR/NIT never force another round" in profile.instructions


@pytest.mark.parametrize(
    "name", ["reviewer", "qa-tester", "ux-reviewer", "designer", "copy-reviewer", "architect"]
)
def test_read_only_roles_reuse_the_manager_owned_pr_worktree(name):
    # Readers clone/install by default was measured filling the shared disk;
    # the default is reuse, and only a conflicting commit or an isolated build
    # collision may justify another checkout.
    profile = load_seed(name)
    assert profile is not None
    body = " ".join(profile.instructions.split())
    assert "manager-owned PR worktree" in body
    assert "no own clone/install" in body or "no clone, install or new worktree" in body
    if name != "architect":
        assert "conflicting commit or isolated build collision" in body
        assert "guide://delivery" in body


@pytest.mark.parametrize("name", ["manager", "coder"])
def test_owning_roles_hold_one_pr_worktree_lifecycle(name):
    profile = load_seed(name)
    assert profile is not None
    body = " ".join(profile.instructions.split())
    assert "ONE PR worktree" in body and "origin/main" in body
    assert "merges or closes" in body or "merges/closes" in body


def test_delivery_guide_owns_the_shared_worktree_rule():
    policy = " ".join((ROOT / "local_operator/guides/delivery/GUIDE.md").read_text().split())
    for required in (
        "owns ONE PR worktree",
        "no clone, install or new worktree",
        "conflicting commit author",
        "isolated mutable build collision",
        "no full-tree copies into scratch",
        "isolated HOME/config",
    ):
        assert required in policy


def test_reader_worktrees_are_sited_and_removed_and_receipts_committed():
    # 2026-11-13 hygiene receipts: reader checkouts left behind, 33 worktrees in
    # session scratchpads and superseded receipts left `??` after merge. Every
    # reader-facing surface names the site and the creator's removal duty.
    policy = " ".join((ROOT / "local_operator/guides/delivery/GUIDE.md").read_text().split())
    for required in (
        "detached-SHA worktree in `<repo>/.worktrees/` (never a scratchpad or /tmp)",
        "its creator removes it before reporting",
        "a merged tree is `git status` clean",
    ):
        assert required in policy, required
    assert "`<repo>/.worktrees/`, never a scratchpad" in DELIVERY_POINTER
    assert "remove your own when the round ends" in DELIVERY_POINTER
    for name in ("reviewer", "qa-tester", "designer", "ux-reviewer"):
        seed = load_seed(name)
        assert seed is not None
        body = " ".join(seed.instructions.split())
        assert "`.worktrees/`, never a scratchpad" in body, name
        assert "remove your own when the round ends" in body, name


def test_stale_copy_reviewer_gets_the_pointer_with_worktree_rule(tmp_path):
    registry = AgentRegistry(tmp_path / "config")
    row = registry.create_agent(_fields(name="copy-reviewer", tags=["role"]))
    registry.set_agent_system_prompt(row.id, "Old copy rules.")
    _, profile = _child_prompt(tmp_path, registry, "copy-reviewer")
    assert profile is not None and profile.instructions.endswith(DELIVERY_POINTER)
    assert "no clone, install or new worktree" in DELIVERY_POINTER
    assert registry.get_agent_system_prompt(row.id) == "Old copy rules."


def test_qa_round_two_retests_only_failed_and_reimpacted_rows():
    # Measured: QA round 2+ re-ran a median 66% of round-1 rows. The seed
    # states the delta rule, keeps real majors blocking, and reuses one tree.
    seed = load_seed("qa-tester")
    assert seed is not None
    body = " ".join(seed.instructions.split())
    for required in (
        "Round 1: the full in-scope QA matrix",
        "re-test only prior FAIL/BLOCKED rows",
        "`retained (<sha>)`",
        "Real BLOCKER/MAJOR still block",
        "never move it to the base",
        "HEAD == briefed SHA",
    ):
        assert required in body


def test_delivery_guide_carries_brief_field_line_and_clean_stream_retention():
    policy = " ".join((ROOT / "local_operator/guides/delivery/GUIDE.md").read_text().split())
    for required in (
        "`Worktree: <abs path> @ <sha> · Round: <N> · Scope: <prev>..<head> · "
        "Retest: <FAIL ids>|none`",
        "A clean stream is NOT re-run when the delta misses its surface",
        "no QA/design/UX round",
        "never sleep-poll it mid-cycle",
        "PYTHONPATH=<worktree>",
    ):
        assert required in policy


@pytest.mark.parametrize("name", ["reviewer", "qa-tester", "designer", "ux-reviewer", "manager"])
def test_roles_name_the_worktree_brief_field(name):
    profile = load_seed(name)
    assert profile is not None and "Worktree:" in profile.instructions


def test_reader_launch_without_worktree_field_notices_the_parent_only(tmp_path):
    from local_operator.harness.subagent import (
        MISSING_WORKTREE_NOTICE,
        missing_worktree_notice,
    )

    assert missing_worktree_notice("qa-tester", "Test PR #1") == MISSING_WORKTREE_NOTICE
    # A named but nonexistent path is the same omission.
    gone = tmp_path / "gone"
    unresolved = missing_worktree_notice("Reviewer", f"Worktree: {gone} @ abc")
    assert unresolved is not None and str(gone) in unresolved
    brief = f"Request: x\nWorktree: {tmp_path} @ abc1234 · Round: 2 · Scope: a..b"
    assert missing_worktree_notice("reviewer", brief) is None
    # The field is found wherever and however managers write it.
    for brief in (
        f"PR: #1 · Worktree: {tmp_path} @ abc",
        f"Worktree: `{tmp_path}` @ abc",
        f"`Worktree: {tmp_path} @ abc`",
        f"**Worktree:** {tmp_path} @ abc",
        f"- Worktree: {tmp_path} @ abc",
    ):
        assert missing_worktree_notice("qa-tester", brief) is None, brief


def test_worktree_field_expands_home(tmp_path, monkeypatch):
    from local_operator.harness.subagent import missing_worktree_notice

    (tmp_path / "w").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert missing_worktree_notice("reviewer", "PR: #1 · Worktree: ~/w @ abc") is None
    # Writers and plain tasks own their tree; no notice.
    assert missing_worktree_notice("coder", "no field") is None
    assert missing_worktree_notice("task", "no field") is None


@pytest.mark.asyncio
async def test_task_receipt_carries_worktree_notice_for_unbriefed_reader(tmp_path):
    from local_operator.harness.jobs import AsyncJobManager
    from local_operator.harness.subagent import MISSING_WORKTREE_NOTICE
    from local_operator.harness.types import ToolContext
    from local_operator.tools import builtin

    prompts: list[str] = []

    def launcher(label, prompt, *, agent="task", effort=None, model=None):
        prompts.append(prompt)
        return f"job-{len(prompts)}"

    context = ToolContext(
        cwd=str(tmp_path), session_id="s", subagent_launcher=launcher, jobs=AsyncJobManager()
    )
    result = await builtin.execute_task(
        "t1",
        {
            "tasks": [
                {"label": "QA", "prompt": "Test PR #9", "agent": "qa-tester"},
                {"label": "Rev", "prompt": f"Worktree: {tmp_path} @ abc", "agent": "reviewer"},
            ]
        },
        None,
        None,
        context,
    )
    assert result.is_error is False
    assert result.text.count(MISSING_WORKTREE_NOTICE) == 1
    # Advisory only: both launched, and the child never sees the notice.
    assert len(prompts) == 2 and all(MISSING_WORKTREE_NOTICE not in p for p in prompts)


def test_attached_stale_registered_role_and_team_manager_get_delivery_pointer(tmp_path):
    """The child-launch pointer used to miss ``/agent`` attach and a team's
    manager session; both resolve through the session's one resolver."""
    from tests.unit.session.test_session import ScriptedStream, make_session

    registry = AgentRegistry(tmp_path / "config")
    for name in ("reviewer", "manager"):
        row = registry.create_agent(_fields(name=name, tags=["role"]))
        registry.set_agent_system_prompt(row.id, f"Old {name} rules.")
    session = make_session(tmp_path, ScriptedStream([]), agent_registry=registry)
    assert session.attach_agent_profile("reviewer") == "reviewer"
    assert DELIVERY_POINTER in session._goal_state.agent_brief
    reviewer = registry.get_agent_by_name("reviewer")
    assert reviewer is not None
    assert registry.get_agent_system_prompt(reviewer.id) == "Old reviewer rules."
    team = Team(
        id="12345678-1234-1234-1234-123456789013",
        name="stale-team",
        manager="manager",
        created_date=datetime.now(timezone.utc),
    )
    session.attach_team(team)
    brief = session._goal_state.team_brief
    assert "Old manager rules." in brief and brief.count(DELIVERY_POINTER) == 1


def test_delivery_guide_carries_the_review_round_economy_rules():
    """Rules that stop review machinery generating rounds after product is done.

    Source data: local-operator-mobile PR #156 — 40 numbered rounds for a
    product clean at round 3, 36 failed native attempts, 20 of them after the
    last product-code commit. Each figure below is measured there, so the
    instruction cannot drift into an invented cap (its own rule 2).
    """
    policy = " ".join((ROOT / "local_operator/guides/delivery/GUIDE.md").read_text().split())
    for required in (
        # rule 1 — evidence budget with an exit
        "3 failed-or-cancelled attempts OR 6 h",
        "41.4 h",
        "extend by a stated amount for a stated reason",
        # rule 2 — measure before cap, name the invariant
        "Measure before you cap; name the invariant",
        "capped hierarchy reads at 20 s over a measured 66–173 s",
        "no check, no shortcut",
        # rule 3 — the harness-only carve-out and its proof
        "scripts/harness_delta_proof.py R H",
        "no byte the shipping pipeline consumes differs between R and H",
        "### Harness delta check — R..H",
        "`.github/harness-paths` present and unchanged",
        # rule 4 — the measured split trigger
        "`--split` prints both",
        "(2,787 vs 2,764 lines)",
        # rule 5 — converge the base once, at the merge gate
        "Converge the base once, at the merge gate",
        "13 times",
        # the sentence the carve-out must not contradict
        "`tests/` is product here",
        "a reachability classifier made safe by failing closed",
        "secrets, signing or publishing",
        "3 failed-or-cancelled attempts OR 6 h",
    ):
        assert required in policy, required


def test_agents_md_admits_exactly_one_more_freshness_exception():
    """AGENTS.md stays the authority: the harness-only delta is the second
    exception to SHA freshness, named in one paragraph and pointed at the guide."""
    policy = " ".join((ROOT / "AGENTS.md").read_text().split())
    for required in (
        "The second exception is a **harness-only delta**",
        "`python scripts/harness_delta_proof.py R H`",
        "`### Harness delta check — R..H`",
        "PR #156 (local-operator-mobile)",
        "`tests/` is product in that manifest",
        "`guide://delivery`",
    ):
        assert required in policy, required
