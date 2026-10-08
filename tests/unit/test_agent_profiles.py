"""Agent role profiles: seeds, registry resolution, and the tool surface.

The behaviour under test is the CONTRACT of a role, never the prose of a
particular seed: the seed bodies are editable operator-facing files, so a test
that pinned their wording would turn every improvement to the guidance into a
test failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from local_operator.agent_profiles import (
    MAX_INSTRUCTIONS_CHARS,
    READ_ONLY_NETWORK_TOOLS,
    READ_ONLY_TOOLS,
    SEED_SHA256_PREFIX,
    SEED_VERSION_PREFIX,
    SEEDS_DIR,
    AgentProfile,
    _split_frontmatter,
    filter_tools,
    install_seed,
    list_seeds,
    load_seed,
    load_seed_version,
    resolve_profile,
    seed_fingerprint,
    seed_tags,
    sync_installed_seeds,
)
from local_operator.agents import AgentEditFields, AgentRegistry


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


ALL_TOOLS: list[_Tool] = [
    _Tool(name)
    for name in (
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "eval",
        "todo",
        "browser",
        "web_search",
        "web_fetch",
    )
]


def seed(name: str) -> AgentProfile:
    """A packaged seed that must exist; keeps the type checker (and the reader)
    from having to reason about None at every call site."""
    profile = load_seed(name)
    assert profile is not None, f"packaged seed {name} is missing"
    return profile


def _edit_fields(**overrides: Any):
    """``AgentEditFields`` with every field spelled out (it is validated in
    strict mode), overridden by the few a test cares about."""
    from local_operator.agents import AgentEditFields

    base: dict[str, Any] = dict(
        name=None,
        label=None,
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


def test_the_packaged_starters_are_all_loadable() -> None:
    """A seed that cannot parse would be invisible at exactly the moment a
    delegation asked for it, so every packaged file is checked here."""
    names = list_seeds()
    assert {"reviewer", "coder", "architect", "manager", "designer", "scout"} <= set(names)
    for name in names:
        profile = load_seed(name)
        assert profile is not None, name
        assert profile.name == name
        assert profile.description, f"{name} has no routing description"
        assert profile.when_to_use, f"{name} does not say when it applies"
        assert profile.instructions.strip(), f"{name} has no guidance"


def test_a_seed_name_cannot_escape_the_catalogue() -> None:
    """The name is resolved against the catalogue, never joined onto a path."""
    assert load_seed("../../etc/passwd") is None
    assert load_seed("") is None
    assert load_seed("does-not-exist") is None


def test_the_reviewer_cannot_edit_but_can_run_the_tests() -> None:
    """The role's whole point: a reviewer that could edit would end up
    reviewing its own patch, and one that could not run anything would file
    findings it never verified."""
    names = {tool.name for tool in filter_tools(ALL_TOOLS, seed("reviewer"))}
    assert "edit" not in names and "write" not in names
    assert "bash" in names and "read" in names


def test_a_read_only_role_changes_nothing_but_can_still_reach_the_web() -> None:
    """Read-only is a promise about CHANGE, not about reach.

    A scout whose surface omitted the network tools reported "I have no
    network access in this session" and fell back to grepping the local disk
    for facts it had been asked to find on the web. Retrieval mutates no more
    than reading a file does, so it belongs in a surface that is defined by
    making no local change — while ``bash``, ``eval``, ``browser``, ``write``
    and ``edit`` stay out by name, tier check or not."""
    names = {tool.name for tool in filter_tools(ALL_TOOLS, seed("scout"))}
    assert names <= {"read", "glob", "grep", "web_search", "web_fetch"}
    assert {"web_search", "web_fetch"} <= names
    assert not names & {"bash", "eval", "browser", "write", "edit"}


def test_every_allowlisted_seed_carries_the_read_only_network_tools() -> None:
    """The three specifications of the read-only surface — ``READ_ONLY_TOOLS``,
    the scout fallback allowlist, and the seed frontmatter — must agree, and
    they are three separate files that drifted before. A seed that restricts
    tools at all is a role that would otherwise be silently offline."""
    for name in list_seeds():
        profile = seed(name)
        if not profile.tools:
            continue  # no allowlist: the role already has the full inventory
        assert set(READ_ONLY_NETWORK_TOOLS) <= set(profile.tools), name


def test_the_scout_fallback_allowlist_matches_the_read_only_surface() -> None:
    """``SCOUT_TOOL_ALLOWLIST`` is the no-profile SAFETY fallback (a stripped
    install, an unreadable registry), so it must still exist — and it must not
    be a second hand-maintained copy that can disagree with the constant, which
    is how the packaged seed and the fallback came to differ on the network
    tools in the first place."""
    from local_operator.harness.subagent import SCOUT_TOOL_ALLOWLIST

    assert SCOUT_TOOL_ALLOWLIST == frozenset(READ_ONLY_TOOLS)
    assert set(READ_ONLY_NETWORK_TOOLS) <= SCOUT_TOOL_ALLOWLIST
    assert SCOUT_TOOL_ALLOWLIST.isdisjoint({"bash", "eval", "browser", "write", "edit"})


def test_a_role_without_an_allowlist_keeps_the_full_inventory() -> None:
    coder = seed("coder")
    assert coder.tools is None
    assert filter_tools(ALL_TOOLS, coder) == ALL_TOOLS
    assert filter_tools(ALL_TOOLS, None) == ALL_TOOLS


def test_the_hands_on_roles_are_told_to_look_things_up() -> None:
    """``coder`` and ``designer`` hit the two cases the general principle is
    weakest on, so each seed carries the role-specific application.

    ``coder`` meets third-party error messages and unfamiliar APIs; ``designer``
    judges surfaces where current practice is the reference. Both seeds had
    zero mention of the web, and neither declares a ``tools:`` allowlist, so
    they already HAD the tools and simply were never told when to use them.
    Contract, not wording.
    """
    coder = seed("coder").instructions
    assert "web_search" in coder
    # A found answer is a lead to verify here, never a patch to paste.
    assert "not a patch" in " ".join(coder.split())

    designer = seed("designer").instructions
    assert "web_search" in designer
    # The guard specific to this role: research informs judgement, but a
    # D-finding must still be visible in the frame, per the seed's own rule
    # that a UI is never reviewed from source alone.
    assert "never as grounds for a finding you cannot see in the" in " ".join(designer.split())


def test_the_hands_on_roles_keep_the_full_inventory() -> None:
    """Neither seed may grow a ``tools:`` line to "enable" the web tools.

    ``tools=None`` means "whatever the parent would build", which already
    includes ``web_search``/``web_fetch`` from ``DEFAULT_TOOL_NAMES``. Adding
    an allowlist to advertise them would RESTRICT the child to exactly that
    list — a capability regression wearing the costume of an enablement.
    """
    for name in ("coder", "designer"):
        assert seed(name).tools is None, name
        assert filter_tools(ALL_TOOLS, seed(name)) == ALL_TOOLS, name


def test_aida_goes_by_the_name_the_operator_configured() -> None:
    """Her display name is the operator's (`aida.name`, default Aida): she
    introduces and refers to herself by their name for her, while the `/aida`
    command keeps its own name — a hard-baked name drifts from the configured
    one the moment the operator renames her."""
    text = seed("aida").instructions
    assert "aida.name" in text
    assert "/aida" in text


def test_aida_hands_off_work_a_team_should_own() -> None:
    """The manager handoff: refresh the project, spawn the manager session
    with the brief, let the manager drive it, and say the handoff out loud —
    who owns it and when she will look again. Without these beats she sits in
    the middle of work that a manager session should carry."""
    flat = " ".join(seed("aida").instructions.split())
    assert "spawn the manager session with the brief" in flat
    assert "who owns it now" in flat


def test_aida_names_the_approval_route_a_headless_delegation_needs() -> None:
    """QA round 1 (advisory): a headless spawn has no terminal to approve its
    tool calls, so without a route the child lands read-only ("Approval
    unavailable … Run with --yolo") and the delegation stalls one recovery
    cycle in. The delegation line must name the launcher's routes and say the
    consequence of having none."""
    flat = " ".join(seed("aida").instructions.split())
    assert "their approvals need a route" in flat
    assert "--control" in flat and "--yolo" in flat and "--tools" in flat
    assert "without one, the run is read-only" in flat


def test_aida_routes_a_watch_to_the_session_that_owns_it() -> None:
    """Her escalation tray is for HER cadence only: a watch that belongs to
    someone else's work goes to a session with its own `wake` — stacking it
    on her timetable leaves the owning session unaware the watch exists."""
    flat = " ".join(seed("aida").instructions.split())
    assert "not another line in your calendar" in flat
    assert "a session with its own `wake`" in flat


def test_aida_stewards_stalled_sessions_stale_projects_and_pending_asks() -> None:
    """The check-in beats the revision added: a stalled session gets one
    bounded wake, a stale project is noticed, and a pending ask from another
    session is answered directly — or, for an operator's decision, surfaced to
    them, never answered on their behalf."""
    flat = " ".join(seed("aida").instructions.split())
    assert "stalled on unfinished work" in flat
    assert "what is going stale" in flat
    assert "pending asks" in flat
    assert "answer status, delegation and routing questions directly" in flat
    assert "is surfaced, not answered" in flat


def test_aida_learns_at_the_narrowest_scope_and_says_so() -> None:
    """The learning loop: each kind of learning names its destination (the
    operator's system prompt, a team's briefs, one agent's own prompt), edits
    the files an EXISTING agent or team actually reads — seeds and templates
    only shape future creations — measures the token cost of what it adds,
    and proposes anything sweeping rather than applying it."""
    text = seed("aida").instructions
    flat = " ".join(text.split())
    assert "Learning and continuous improvement" in text
    assert "system_prompt.md" in text
    assert "teams/<id>/instructions.md" in text
    assert "agents/<id>/system_prompt.md" in text
    assert "never overwritten without an explicit force" in flat
    assert "reaches a live copy only when a sync runs" in flat
    assert "token cost" in flat
    assert "proposed, not applied" in flat


def test_aidas_seed_reaches_the_loader_whole() -> None:
    """The deploy guard for the revision: the loader truncates a body past
    ``MAX_INSTRUCTIONS_CHARS`` (and the manifest generator refuses it), so a
    seed must fit its budget un-truncated — this turns the next over-budget
    addition into a red test instead of silently shipped-short guidance."""
    text = (SEEDS_DIR / "aida.md").read_text(encoding="utf-8")
    _meta, body = _split_frontmatter(text)
    profile = seed("aida")
    assert len(body) <= MAX_INSTRUCTIONS_CHARS
    assert len(profile.instructions) == len(body), "the loader must not truncate her seed"


def test_an_allowlist_naming_absent_tools_matches_nothing_rather_than_raising() -> None:
    """A profile written on another machine (or naming an MCP tool this
    session never loaded) must still run, with the tools it does have."""
    profile = AgentProfile(name="x", tools=("read", "mcp__elsewhere__thing"))
    assert [tool.name for tool in filter_tools(ALL_TOOLS, profile)] == ["read"]


def test_only_the_delegating_roles_may_delegate() -> None:
    """A reviewer spawning children turns one review into an unwatched
    fan-out; a manager coordinating is the case that legitimately needs it."""
    assert seed("manager").may_delegate is True
    for name in ("reviewer", "coder", "scout", "architect", "designer"):
        assert seed(name).may_delegate is False, name


def test_the_preamble_is_empty_when_a_role_says_nothing() -> None:
    """It rides in front of every prompt, so a role with no guidance must
    cost nothing rather than emitting a header with nothing under it."""
    assert AgentProfile(name="bare").preamble == ""
    assert seed("reviewer").preamble.startswith("[role: reviewer]")


def test_instructions_are_bounded(tmp_path) -> None:
    """A profile is user data prepended to every turn of every run of the
    role, so an unbounded body would be an unbounded per-turn bill."""
    registry = AgentRegistry(tmp_path)
    agent = registry.create_agent(_edit_fields(name="verbose", tags=["role"]))
    registry.set_agent_system_prompt(agent.id, "x" * (MAX_INSTRUCTIONS_CHARS * 2))
    profile = resolve_profile("verbose", registry=registry)
    assert profile is not None
    assert len(profile.instructions) == MAX_INSTRUCTIONS_CHARS


class TestResolution:
    def test_task_never_resolves_a_profile(self, tmp_path) -> None:
        """The common launch must pay no registry lookup at all."""
        assert resolve_profile("task", registry=AgentRegistry(tmp_path)) is None
        assert resolve_profile(None) is None
        assert resolve_profile("") is None

    def test_an_unknown_role_resolves_to_nothing(self, tmp_path) -> None:
        """The caller degrades to a full child; a typo must not lose the work."""
        assert resolve_profile("no-such-role", registry=AgentRegistry(tmp_path)) is None

    def test_a_packaged_starter_resolves_without_being_installed(self) -> None:
        profile = resolve_profile("reviewer")
        assert profile is not None and profile.agent_id is None

    def test_the_operators_own_profile_wins_over_the_starter(self, tmp_path) -> None:
        """Once an operator has a reviewer of their own, theirs is the one that
        runs — otherwise editing the guidance would have no effect."""
        registry = AgentRegistry(tmp_path)
        installed = install_seed("reviewer", registry=registry)
        assert installed is not None
        installed = installed[0]
        registry.set_agent_system_prompt(str(installed.agent_id), "MY HOUSE RULES")

        profile = resolve_profile("reviewer", registry=registry)
        assert profile is not None
        assert profile.instructions == "MY HOUSE RULES"
        assert profile.agent_id == installed.agent_id

    def test_a_broken_registry_falls_back_instead_of_failing(self) -> None:
        """Role guidance is enrichment; losing the delegation over a registry
        problem would be a worse outcome than running without the role."""

        class Broken:
            def get_agent_by_name(self, name):  # noqa: ANN001
                raise RuntimeError("registry on fire")

        profile = resolve_profile("reviewer", registry=Broken())
        assert profile is not None and profile.agent_id is None


class TestInstall:
    def test_installing_makes_an_ordinary_editable_registry_row(self, tmp_path) -> None:
        registry = AgentRegistry(tmp_path)
        result = install_seed("reviewer", registry=registry)
        assert result is not None
        profile, already_installed = result
        assert profile.agent_id and not already_installed
        agent = registry.get_agent_by_name("reviewer")
        assert agent is not None
        assert "role" in agent.tags
        assert registry.get_agent_system_prompt(agent.id).strip()

    def test_installing_twice_neither_duplicates_nor_clobbers(self, tmp_path) -> None:
        """Two launches of the same role can race; the second must not undo an
        edit the operator made to the first."""
        registry = AgentRegistry(tmp_path)
        first_result = install_seed("reviewer", registry=registry)
        assert first_result is not None
        first, first_already_installed = first_result
        assert not first_already_installed
        registry.set_agent_system_prompt(str(first.agent_id), "EDITED")

        second_result = install_seed("reviewer", registry=registry)
        assert second_result is not None
        second, second_already_installed = second_result
        assert second_already_installed, "a second install is a deliberate no-op and must say so"
        assert second.agent_id == first.agent_id
        assert second.instructions == "EDITED"
        assert len([a for a in registry.list_agents() if a.name == "reviewer"]) == 1

    def test_an_unknown_starter_installs_nothing(self, tmp_path) -> None:
        assert install_seed("nope", registry=AgentRegistry(tmp_path)) is None

    def test_the_role_fields_survive_a_round_trip(self, tmp_path) -> None:
        """Tools/effort/delegate are encoded in tags because AgentData is a
        persisted, API-exposed model; this is the guard that the encoding and
        the decoding still agree."""
        registry = AgentRegistry(tmp_path)
        install_seed("manager", registry=registry)
        packaged = seed("manager")
        profile = resolve_profile("manager", registry=registry)
        assert profile is not None
        assert profile.tools == packaged.tools
        assert profile.effort == packaged.effort
        assert profile.may_delegate == packaged.may_delegate


def test_seed_tags_encode_only_what_is_set() -> None:
    """Every SET field is encoded — and the class is always one of them.

    The class is the exception to "only what is set" on purpose: an absent tag
    has to mean "this row was never classified" (it is the state every install
    from before the class feature is in, and the repair for those rows keys on
    it), so a profile that is reactive must say so. Leaving it out is how an
    ordinary role edit — which rebuilds a row's tags from its profile — used to
    strip a deliberate ``class:reactive`` and let the repair silently re-arm a
    check-in the operator had switched off (agent review round 1, R1).
    """
    assert seed_tags(AgentProfile(name="plain")) == ("role", "class:reactive")
    tags = seed_tags(AgentProfile(name="x", tools=("read",), effort="lo", may_delegate=True))
    assert set(tags) == {"role", "class:reactive", "tools:read", "effort:lo", "delegate:yes"}


@pytest.mark.parametrize("spelling", ["delegate", "may_delegate"])
def test_both_delegate_spellings_are_accepted(spelling: str, tmp_path) -> None:
    """A human editing a seed should not have to remember which key the parser
    happens to prefer."""
    from local_operator.agent_profiles import _profile_from_text

    text = f"---\nname: x\ndescription: d\n{spelling}: yes\n---\nbody"
    assert _profile_from_text("x", text).may_delegate is True


# ---------------------------------------------------------------------------
# Round-1 review regressions on resolution itself.
# ---------------------------------------------------------------------------


def test_a_same_named_non_role_agent_does_not_become_the_role(tmp_path) -> None:
    """C2: `get_agent_by_name` searches a flat namespace shared with ordinary
    chat agents, so without the role tag an agent merely CALLED `reviewer` was
    launched as one — with no allowlist, i.e. the full write inventory, while
    the child was still told it was a reviewer."""
    registry = AgentRegistry(tmp_path)
    agent = registry.create_agent(_edit_fields(name="reviewer", description="my chat agent"))
    registry.set_agent_system_prompt(agent.id, "Be agreeable.")

    profile = resolve_profile("reviewer", registry=registry)

    assert profile is not None
    assert profile.agent_id is None, "must fall through to the packaged seed"
    assert profile.tools, "a role without an allowlist is the fail-open case"
    assert "Be agreeable." not in profile.instructions


def test_a_role_tagged_agent_is_still_honoured(tmp_path) -> None:
    """The guard must not break the feature it protects."""
    registry = AgentRegistry(tmp_path)
    agent = registry.create_agent(
        _edit_fields(name="reviewer", description="house", tags=["role", "tools:read"])
    )
    registry.set_agent_system_prompt(agent.id, "HOUSE RULES")

    profile = resolve_profile("reviewer", registry=registry)

    assert profile is not None and profile.agent_id == agent.id
    assert profile.tools == ("read",)
    assert profile.instructions == "HOUSE RULES"


def test_installing_over_a_non_role_name_refuses_rather_than_lying(tmp_path) -> None:
    """C4 at the source: it used to return the existing row, which reads as a
    successful install to every caller."""
    import pytest as _pytest

    from local_operator.agent_profiles import NameTakenError

    registry = AgentRegistry(tmp_path)
    registry.create_agent(_edit_fields(name="reviewer", description="chat"))
    with _pytest.raises(NameTakenError):
        install_seed("reviewer", registry=registry)


def test_role_lookup_folds_case_like_the_seed_lookup_does(tmp_path) -> None:
    """C9: seeds fold case and the registry did not, so `agent="Reviewer"`
    found the PACKAGED seed while ignoring the operator's own."""
    registry = AgentRegistry(tmp_path)
    agent = registry.create_agent(_edit_fields(name="Reviewer", description="house", tags=["role"]))
    registry.set_agent_system_prompt(agent.id, "HOUSE RULES")

    for spelling in ("Reviewer", "reviewer", "REVIEWER"):
        profile = resolve_profile(spelling, registry=registry)
        assert profile is not None, spelling
        assert profile.instructions == "HOUSE RULES", spelling


def test_an_exact_match_still_wins_over_a_case_folded_one(tmp_path) -> None:
    """The fold is a fallback, not a replacement: it runs only when the exact
    lookup found nothing."""
    registry = AgentRegistry(tmp_path)
    exact = registry.create_agent(_edit_fields(name="triage", description="d", tags=["role"]))
    registry.set_agent_system_prompt(exact.id, "EXACT")
    other = registry.create_agent(_edit_fields(name="TRIAGE", description="d", tags=["role"]))
    registry.set_agent_system_prompt(other.id, "FOLDED")

    profile = resolve_profile("triage", registry=registry)
    assert profile is not None and profile.instructions == "EXACT"


def test_a_non_role_exact_match_does_not_shadow_the_operators_own_role(tmp_path) -> None:
    """C11: the round-1 fold fix stopped at the exact lookup, so a non-role row
    matching exactly discarded the hit and the fold never ran — reopening the
    very bug the fold was added for, in the one arrangement its test missed."""
    registry = AgentRegistry(tmp_path)
    registry.create_agent(_edit_fields(name="reviewer", description="my chat agent"))
    role = registry.create_agent(
        _edit_fields(name="Reviewer", description="house", tags=["role", "tools:read"])
    )
    registry.set_agent_system_prompt(role.id, "HOUSE RULES")

    profile = resolve_profile("reviewer", registry=registry)

    assert profile is not None
    assert profile.agent_id == role.id, "the operator's own role must win"
    assert profile.instructions == "HOUSE RULES"


# -- update checks: sync_installed_seeds -------------------------------------
#
# The package is the thing that MOVES in these tests, so each one rewrites a
# scratch copy of the seed catalogue and redirects ``SEEDS_DIR`` at it. The
# verdict table is exercised against that moved package, never against a
# hand-stamped row pretending to be one — the stamps installed here are the
# ones install_seed itself writes.


@pytest.fixture()
def scratch_seeds(tmp_path, monkeypatch) -> Path:
    """A writable copy of the packaged seeds, so a test can move the package."""

    import local_operator.agent_profiles as agent_profiles

    destination = tmp_path / "agent_seeds"
    shutil.copytree(Path(agent_profiles.SEEDS_DIR), destination)
    monkeypatch.setattr(agent_profiles, "SEEDS_DIR", destination)
    return destination


def move_seed(seeds_dir: Path, name: str, *, version: str, body: str | None = None) -> None:
    """Rewrite one packaged seed: a new ``version:`` and new body text.

    Both move together because that is the update sync exists for: a body
    change ships with a version bump (the manifest's byte-identity test owns
    that rule), and sync's whole verdict table is defined over the pair.
    """

    path = seeds_dir / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) == 3, f"{name}.md has no frontmatter"
    parts[1] = re.sub(r"^version:.*$", f"version: {version}", parts[1], flags=re.M)
    parts[2] = f"\n\n{body or f'Guidance for {name} as of {version}.'}\n"
    path.write_text("---".join(parts), encoding="utf-8")


def _install(registry: AgentRegistry, name: str = "reviewer"):
    installed = install_seed(name, registry=registry)
    assert installed is not None
    row = registry.get_agent_by_name(name)
    assert row is not None
    return row


def test_an_untouched_copy_of_an_unmoved_starter_is_up_to_date(scratch_seeds, tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    before = list(row.tags)

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "up-to-date"
    assert verdict.applied is False
    assert verdict.installed_version == verdict.packaged_version
    after = registry.get_agent_by_name("reviewer")
    assert after is not None
    assert list(after.tags) == before


def test_an_untouched_copy_updates_in_place_when_the_starter_moves(scratch_seeds, tmp_path) -> None:
    """The headline path: update local-operator, run sync, the starter updates.

    This is exactly the case a version+divergence-only check cannot classify
    (the row differs from the NEW packaged text whether or not it was edited),
    which is why install records a fingerprint of what it wrote.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    before_prompt = registry.get_agent_system_prompt(row.id)
    move_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean"
    assert verdict.applied is True
    # Stamped from the packaged reviewer at install: the seed moved to 1.2.0
    # in the 2026-10-06 reporting-rule revision (was 1.1.0).
    assert verdict.installed_version == "1.2.0"
    assert verdict.packaged_version == "2.0.0"
    assert verdict.diverged_fields == ("instructions",)
    # The echo: recoverable by copy-paste, the same guarantee reset gives.
    assert verdict.replaced_instructions == before_prompt
    after = registry.get_agent_by_name("reviewer")
    assert after is not None
    assert registry.get_agent_system_prompt(after.id).strip() == "REVIEWER v2 GUIDANCE"
    assert "seed_version:2.0.0" in after.tags
    # A second run has nothing left to do — and does not re-echo.
    (again,) = sync_installed_seeds(registry)
    assert again.verdict == "up-to-date"


def test_an_edited_copy_refuses_and_force_replaces_it(scratch_seeds, tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    registry.set_agent_system_prompt(row.id, "MY EDITED PROMPT")
    move_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-diverged"
    assert verdict.applied is False
    assert verdict.diverged_fields == ("instructions",)
    # The remedy names the flag pair the CLI actually honours: "re-run with
    # force" sent users to a hidden deprecated flag whose own warning points at
    # --replace (finding F2 of #2060).
    assert "--replace --yes" in verdict.detail
    assert "force" not in verdict.detail
    # The refusal is a real one: the edit is still there.
    assert registry.get_agent_system_prompt(row.id) == "MY EDITED PROMPT"

    (forced,) = sync_installed_seeds(registry, force=True)

    assert forced.verdict == "outdated-diverged"
    assert forced.applied is True
    assert forced.replaced_instructions == "MY EDITED PROMPT"
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"


def test_a_current_but_edited_copy_is_not_a_sync_target(scratch_seeds, tmp_path) -> None:
    """The packaged starter has not moved, so there is nothing to PULL.

    Sync must not turn into a silent reset of local edits: it reports the row
    as current (v2's drift is visible in ``show``, which owns that surface)
    and leaves the text alone.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    registry.set_agent_system_prompt(row.id, "MY EDITED PROMPT")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "up-to-date"
    assert "local edits" in verdict.detail
    assert registry.get_agent_system_prompt(row.id) == "MY EDITED PROMPT"


def test_a_row_from_before_the_markers_is_refused_when_its_starter_moved(
    scratch_seeds, tmp_path
) -> None:
    """``seed_origin``'s safe direction, applied to updates.

    An old row has no recorded baseline, so \"unedited\" cannot be proven once
    the packaged text moves; the refusal (force to apply) is the direction that
    cannot destroy a person's work. When the text has NOT moved, the row still
    matches the package and sync leaves it alone.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    # Strip exactly the sync markers, keeping the profile tags a real old row
    # carries (dropping a ``tools:`` tag would be a different scenario — an
    # edit — and would confuse this test with the refusal one above).
    registry.update_agent(
        row.id,
        _edit_fields(
            tags=[
                tag
                for tag in row.tags
                if not tag.startswith((SEED_VERSION_PREFIX, SEED_SHA256_PREFIX))
            ]
        ),
    )

    (still_current,) = sync_installed_seeds(registry)
    assert still_current.verdict == "up-to-date"

    move_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    (verdict,) = sync_installed_seeds(registry)
    assert verdict.verdict == "outdated-diverged"
    assert verdict.applied is False

    (forced,) = sync_installed_seeds(registry, force=True)
    assert forced.applied is True


def _move_seed_body_without_bumping(seeds_dir: Path, name: str, *, body: str) -> None:
    """Rewrite one packaged seed's BODY, leaving ``version:`` alone.

    ``move_seed`` keeps the pair together by convention, but nothing enforces
    that pairing: the generator derives the body's ``instructions_sha256`` and
    the frontmatter's ``version`` as independent fields, so a body shipped
    under a stale version is a real (if careless) package move — and the case
    that used to be mis-classified as a local edit because the versions still
    matched (agent review round 1, M1).
    """

    path = seeds_dir / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) == 3, f"{name}.md has no frontmatter"
    parts[2] = f"\n\n{body}\n"
    path.write_text("---".join(parts), encoding="utf-8")


def test_an_unbumped_body_change_is_still_an_update(scratch_seeds, tmp_path) -> None:
    """Classification is the FINGERPRINT's job, not the version string's.

    Comparing versions first called this move "this copy has local edits": the
    row was reported up to date and was un-updateable even with ``force``
    (agent review round 1, M1).
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    before_prompt = registry.get_agent_system_prompt(row.id)
    _move_seed_body_without_bumping(
        scratch_seeds, "reviewer", body="REVIEWER, SAME VERSION, NEW TEXT"
    )

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean"
    assert verdict.applied is True
    # Both sides read the seed's own version, untouched by the body-only move:
    # 1.2.0 since the 2026-10-06 reporting-rule revision (was 1.1.0).
    assert verdict.installed_version == verdict.packaged_version == "1.2.0"
    assert verdict.replaced_instructions == before_prompt
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER, SAME VERSION, NEW TEXT"


def test_an_unbumped_body_change_refuses_a_local_edit_but_force_applies(
    scratch_seeds, tmp_path
) -> None:
    """The same move with an edit in the way: refusal, then force."""

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    registry.set_agent_system_prompt(row.id, "MY EDITED PROMPT")
    _move_seed_body_without_bumping(
        scratch_seeds, "reviewer", body="REVIEWER, SAME VERSION, NEW TEXT"
    )

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-diverged"
    assert verdict.applied is False
    assert verdict.diverged_fields == ("instructions",)
    assert registry.get_agent_system_prompt(row.id) == "MY EDITED PROMPT"

    (forced,) = sync_installed_seeds(registry, force=True)

    assert forced.applied is True
    assert forced.replaced_instructions == "MY EDITED PROMPT"


def test_a_row_with_no_install_record_says_what_it_cannot_tell(scratch_seeds, tmp_path) -> None:
    """A pre-marker row is evidence of neither a move nor an edit.

    With no recorded baseline both readings produce the same difference, so the
    report must not assert either one — the old wording announced an update
    the row had no way to know about, and pointed at ``force``, which would
    have reverted the user's own text (agent review round 1, M1). The refusal
    itself is unchanged: force is the explicit way to take the packaged text.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    registry.set_agent_system_prompt(row.id, "MY EDITED PROMPT")
    registry.update_agent(
        row.id,
        _edit_fields(
            tags=[
                tag
                for tag in row.tags
                if not tag.startswith((SEED_VERSION_PREFIX, SEED_SHA256_PREFIX))
            ]
        ),
    )

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-diverged"
    assert verdict.applied is False
    assert "no install record" in verdict.detail
    assert "cannot tell" in verdict.detail
    assert registry.get_agent_system_prompt(row.id) == "MY EDITED PROMPT"


def test_a_package_move_backward_still_applies_and_shows_both_versions(
    scratch_seeds, tmp_path
) -> None:
    """The intended direction semantics (QA round 1, Q-1).

    ``sync`` means "make this copy match the starter THIS build ships", so a
    package moving from a newer version back to an older one (a ``lop``
    downgrade, a channel switch, a reverted starter) is the same update in the
    other direction. Both versions and the replaced text ride in the receipt,
    so the move is visible and never silent — pinned here as intended rather
    than incidental.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    move_seed(scratch_seeds, "reviewer", version="1.0.1", body="REVIEWER v1.0.1 GUIDANCE")
    sync_installed_seeds(registry)
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v1.0.1 GUIDANCE"

    move_seed(scratch_seeds, "reviewer", version="0.9.0", body="REVIEWER v0.9.0 GUIDANCE")
    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean"
    assert verdict.applied is True
    assert (verdict.installed_version, verdict.packaged_version) == ("1.0.1", "0.9.0")
    assert verdict.replaced_instructions is not None
    assert verdict.replaced_instructions.strip() == "REVIEWER v1.0.1 GUIDANCE"
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v0.9.0 GUIDANCE"


def test_a_case_renamed_row_is_updated_in_place_not_duplicated(scratch_seeds, tmp_path) -> None:
    """Discovery folds names; the apply must resolve through the same fold.

    The desktop update route can rename a row (``reviewer`` → ``Reviewer``).
    Folded discovery found it, the exact-case install lookup missed it, and
    ``create_agent`` minted a second ``reviewer`` beside it — reported as an
    update while the duplicate appeared (agent review round 1, M2).
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    registry.update_agent(row.id, _edit_fields(name="Reviewer"))
    move_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean"
    assert verdict.applied is True
    names = sorted(agent.name for agent in registry.list_agents())
    assert names == ["Reviewer"], "the rename must survive; no duplicate row"
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"


def test_install_does_not_mint_a_case_duplicate(scratch_seeds, tmp_path) -> None:
    """The same fold on the install verb: a renamed row IS the row.

    ``op='install'`` and ``reset`` run through this lookup; exact-case-only
    meant asking to install ``reviewer`` beside a renamed ``Reviewer`` created
    a rival row holding the packaged text (agent review round 1, M2).
    """

    registry = AgentRegistry(tmp_path / "config")
    role = registry.create_agent(_edit_fields(name="Reviewer", description="house", tags=["role"]))
    registry.set_agent_system_prompt(role.id, "HOUSE RULES")

    installed = install_seed("reviewer", registry=registry)

    assert installed is not None
    names = sorted(agent.name for agent in registry.list_agents())
    assert names == ["Reviewer"], "no second row under the packaged spelling"
    assert (
        registry.get_agent_system_prompt(role.id) == "HOUSE RULES"
    ), "a non-overwriting install writes nothing"


def test_sync_by_name_reports_what_is_not_an_update_target(scratch_seeds, tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "config")

    (missing,) = sync_installed_seeds(registry, names=["reviewer"])
    assert missing.verdict == "not-installed"
    assert "op='install'" in missing.detail
    (unknown,) = sync_installed_seeds(registry, names=["nope"])
    assert unknown.verdict == "not-installed"
    assert "no installed role of this name" in unknown.detail

    # A hand-authored role under a starter's name is NOT a copy of the starter
    # and must never be rewritten: the not-installed verdict says which verb
    # actually applies.
    registry.create_agent(_edit_fields(name="reviewer", description="mine", tags=["role"]))
    (authored,) = sync_installed_seeds(registry, names=["reviewer"])
    assert authored.verdict == "not-installed"
    assert "not installed from a packaged starter" in authored.detail


def test_sync_by_name_only_processes_the_named_seed(scratch_seeds, tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "config")
    _install(registry, "reviewer")
    _install(registry, "coder")

    verdicts = sync_installed_seeds(registry, names=["coder"])

    assert [verdict.name for verdict in verdicts] == ["coder"]


def test_the_fingerprint_covers_the_fields_the_seed_writes() -> None:
    """A baseline that missed a field would call an edit to it \"clean\".

    The allowlist is the one with a scar: a widened ``tools:`` tag is a
    capability change the restore path exists to protect, so a fingerprint
    that hashed only the prose would silently overwrite it.
    """

    seed = load_seed("reviewer")
    assert seed is not None
    baseline = seed_fingerprint(seed)

    widened = AgentProfile(
        name=seed.name,
        description=seed.when_to_use or seed.description,
        instructions=seed.instructions,
        tools=tuple(seed.tools or ()) + ("write",),
        effort=seed.effort,
        may_delegate=seed.may_delegate,
    )
    assert seed_fingerprint(widened) != baseline


def test_seed_version_reads_the_frontmatter_the_manifest_publishes() -> None:
    """One version, two readers: the manifest and the install stamp."""

    for name in list_seeds():
        assert load_seed_version(name), name
    assert load_seed_version("does-not-exist") == ""


# ---------------------------------------------------------------------------
# The proactive class (R29): frontmatter → tags → read-back
# ---------------------------------------------------------------------------


def test_the_class_frontmatter_round_trips_through_seed_tags() -> None:
    from local_operator.action_class import PROACTIVE, REACTIVE
    from local_operator.agent_profiles import _profile_from_text

    proactive = _profile_from_text(
        "companion",
        "---\nname: companion\ndescription: d\nclass: proactive\n---\nhi",
    )
    assert proactive.action_class == PROACTIVE
    assert "class:proactive" in seed_tags(proactive)

    plain = _profile_from_text("plain", "---\nname: plain\ndescription: d\n---\nhi")
    assert plain.action_class == REACTIVE
    # A profile with no ``class:`` frontmatter is reactive, and the encoding SAYS
    # so rather than implying it: see ``test_seed_tags_encode_only_what_is_set``
    # for why the absence carries a different meaning now.
    assert "class:reactive" in seed_tags(plain)


def test_profile_from_agent_reads_the_class_tag_back(tmp_path) -> None:
    from local_operator.action_class import PROACTIVE
    from local_operator.agent_profiles import profile_from_agent

    def fields(**overrides: Any):
        """``AgentEditFields`` with every field spelled out (strict mode)."""
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

    registry = AgentRegistry(tmp_path)
    registry.create_agent(fields(name="steward", tags=["role", "class:proactive"]))
    row = registry.get_agent_by_name("steward")
    assert row is not None
    profile = profile_from_agent(registry, row)
    assert profile.action_class == PROACTIVE

    registry.create_agent(fields(name="plain", tags=["role"]))
    row = registry.get_agent_by_name("plain")
    assert row is not None
    assert profile_from_agent(registry, row).action_class == "reactive"


def test_the_fingerprint_covers_the_class_field() -> None:
    """A baseline that missed the class would call an edit to it "clean".

    The class is a seed-written field (``seed_tags`` encodes it), so the same
    argument the tools-allowlist widening scar settled applies: a fingerprint
    that hashed only the prose would let a class flip read as untouched.
    """
    from local_operator.action_class import PROACTIVE, REACTIVE

    seed = load_seed("reviewer")
    assert seed is not None
    baseline = seed_fingerprint(seed)
    flipped = AgentProfile(
        name=seed.name,
        description=seed.when_to_use or seed.description,
        instructions=seed.instructions,
        tools=tuple(seed.tools) if seed.tools else None,
        effort=seed.effort,
        may_delegate=seed.may_delegate,
        action_class=PROACTIVE if seed.action_class == REACTIVE else REACTIVE,
    )
    assert seed_fingerprint(flipped) != baseline


# ---------------------------------------------------------------------------
# #2060: the revision ledger, the narrow writer, and the startup pass
# ---------------------------------------------------------------------------


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` by relative path: for byte-level write-nothing cells."""

    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _legacy_fingerprint_of(registry: AgentRegistry, agent: Any) -> str:
    """The v0.63.5..v0.64.8 five-field stamp for a row, computed inline.

    Deliberately NOT ``agent_profiles._legacy_fingerprint``: this cell has to
    keep working - and keep FAILING against the pre-fix tree - as a statement
    about the wire, and the pre-fix tree has no private helper to borrow. The
    formula is the one those releases shipped: ``json.dumps`` of the five
    values with pinned separators, sha256 over the UTF-8 bytes.
    """

    from local_operator.agent_profiles import profile_from_agent

    profile = profile_from_agent(registry, agent)
    values = [
        (profile.instructions or "").strip(),
        (profile.when_to_use or profile.description or "").strip(),
        tuple(profile.tools) if profile.tools else None,
        profile.effort or None,
        bool(profile.may_delegate),
    ]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_published_revision(seeds_dir: Path, name: str) -> None:
    """Append the CURRENT scratch seed text to its scratch ledger.

    The real release flow changes the seed file and regenerates the ledger in
    one commit (``scripts/gen_agent_seed_revisions.py``), so a scratch move
    that skips the ledger is the DRAFT case - exercised deliberately by the
    stamp-fallback cells - and every "a published update applies" cell must
    publish BOTH halves. ``make_seed_revision``/``render_seed_revisions`` are
    the generator's own constructor and serialiser, so a scratch entry cannot
    drift from a generated one.
    """

    import local_operator.agent_profiles as agent_profiles

    revisions = agent_profiles.load_seed_revisions(seeds_dir)
    profile = agent_profiles.load_seed(name)
    assert profile is not None
    version = agent_profiles.load_seed_version(name)
    digest = hashlib.sha256(f"{name}:{version}:{profile.instructions}".encode("utf-8"))
    sha = digest.hexdigest()[:40]
    updated = {key: list(entries) for key, entries in revisions.items()}
    updated.setdefault(name, []).append(
        agent_profiles.make_seed_revision(
            profile,
            sha=sha,
            version=version,
            declared_class=agent_profiles.load_seed_class(name),
        )
    )
    (seeds_dir / agent_profiles.SEED_REVISIONS_NAME).write_text(
        agent_profiles.render_seed_revisions(updated), encoding="utf-8"
    )


def publish_seed(
    seeds_dir: Path,
    name: str,
    *,
    version: str,
    body: str,
    tools: str | None = None,
) -> None:
    """Move the packaged seed AND append the matching ledger entry.

    The one helper every "a published update" cell goes through: it rewrites
    the scratch seed (version + body, optionally the ``tools:`` line) and then
    records the new text in the scratch ledger, exactly as a release does.
    """

    path = seeds_dir / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) == 3, f"{name}.md has no frontmatter"
    parts[1] = re.sub(r"^version:.*$", f"version: {version}", parts[1], flags=re.M)
    if tools is not None:
        if re.search(r"^tools:", parts[1], flags=re.M):
            parts[1] = re.sub(r"^tools:.*$", f"tools: {tools}", parts[1], flags=re.M)
        else:
            parts[1] = parts[1].rstrip("\n") + f"\ntools: {tools}\n"
    parts[2] = f"\n\n{body}\n"
    path.write_text("---".join(parts), encoding="utf-8")
    _append_published_revision(seeds_dir, name)


def test_a_legacy_stamp_row_on_a_published_revision_is_clean_and_applies(
    scratch_seeds, tmp_path
) -> None:
    """#2060 defect 3, in the reporter's exact shape (finding F1).

    v0.63.5-v0.64.8 wrote ``seed_sha256:`` under a five-field formula; the
    class feature (v0.64.9) appended a sixth field and changed EVERY hash, so
    those rows could never again recompute to their stamp - however untouched -
    and a plain sync answered "outdated-diverged". The stamp-FAMILY check
    (either era's formula) re-proves the row: it still holds exactly what was
    installed, so a plain sync applies the update.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    legacy = _legacy_fingerprint_of(registry, row)
    registry.update_agent(
        row.id,
        _edit_fields(
            tags=[tag for tag in row.tags if not tag.startswith(SEED_SHA256_PREFIX)]
            + [f"{SEED_SHA256_PREFIX}{legacy}"]
        ),
    )
    move_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean"
    assert verdict.applied is True
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"


def test_a_row_with_no_install_record_is_re_proven_by_the_ledger(scratch_seeds, tmp_path) -> None:
    """Pre-stamp rows have no fingerprint; the LEDGER is what re-proves them.

    A row whose text is a published revision is clean by construction once the
    ledger can position both ends of the move - the case that used to need
    ``--force`` or a reinstall. The package is PUBLISHED here (ledger entry
    appended), because the proof depends on the packaged text being a real
    revision too: an unpublished draft stays unprovable, by design.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    before_prompt = registry.get_agent_system_prompt(row.id)
    # Strip the install record entirely: a pre-stamp row's exact shape.
    registry.update_agent(
        row.id,
        _edit_fields(
            tags=[
                tag
                for tag in row.tags
                if not tag.startswith((SEED_VERSION_PREFIX, SEED_SHA256_PREFIX))
            ]
        ),
    )
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean"
    assert verdict.applied is True
    assert verdict.behind_by == 1
    assert verdict.replaced_instructions == before_prompt
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"


def test_read_only_sync_reports_available_and_writes_nothing(scratch_seeds, tmp_path) -> None:
    """``apply=False`` is write-nothing, byte-level (the F2 trap).

    ``lop agents sync --dry-run`` used to APPLY clean seed updates while
    promising "Show what would change; write nothing", and a read-only
    classification that still rendered "updated" would be the same over-claim
    on the other side. Both halves are pinned here: the verdict is
    ``outdated-clean`` with ``applied=False`` and the ``behind_by`` distance,
    and every byte under the config dir is untouched.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    before = _tree_bytes(tmp_path)

    (verdict,) = sync_installed_seeds(registry, apply=False)

    assert verdict.verdict == "outdated-clean"
    assert verdict.applied is False
    assert verdict.behind_by == 1
    assert verdict.replaced_instructions is None
    assert _tree_bytes(tmp_path) == before
    assert registry.get_agent_system_prompt(row.id).strip() != "REVIEWER v2 GUIDANCE"


def test_the_narrow_writer_preserves_user_owned_row_data(scratch_seeds, tmp_path) -> None:
    """F4: the old clean apply reset the label and dropped user tags.

    The narrow writer exists for exactly this: an update nobody asked for
    replaces only what the seed owns - instructions, routing description,
    seed-owned tags and the stamps - and leaves the label, the model pin and
    any non-seed tag alone.
    """

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    registry.update_agent(
        row.id,
        _edit_fields(
            label="My Reviewer",
            model="gpt-test-pin",
            tags=[*row.tags, "favourite"],
        ),
    )
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean" and verdict.applied is True
    after = registry.get_agent_by_name("reviewer")
    assert after is not None
    assert after.label == "My Reviewer"
    assert after.model == "gpt-test-pin"
    assert "favourite" in after.tags
    assert "seed_version:2.0.0" in after.tags
    assert registry.get_agent_system_prompt(after.id).strip() == "REVIEWER v2 GUIDANCE"


def test_a_switched_class_survives_an_update_and_reset_restores_it(scratch_seeds, tmp_path) -> None:
    """A switched ``class:`` is user data like the label (ADR Q2).

    The narrow writer preserves a class that matches NO ledger entry for the
    row's identity - the row was deliberately switched - and the packaged
    class reaching a switched row stays ``reset``'s job. Both halves are
    asserted so the trade cannot silently become a loss.
    """

    from local_operator.action_class import PROACTIVE, REACTIVE

    registry = AgentRegistry(tmp_path / "config")
    row = _install(registry)
    assert "class:reactive" in row.tags
    registry.update_agent(
        row.id,
        _edit_fields(
            tags=[tag for tag in row.tags if not tag.lower().startswith("class:")]
            + [f"class:{PROACTIVE}"]
        ),
    )
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "outdated-clean" and verdict.applied is True
    after = registry.get_agent_by_name("reviewer")
    assert after is not None
    assert f"class:{PROACTIVE}" in after.tags
    assert registry.get_agent_system_prompt(after.id).strip() == "REVIEWER v2 GUIDANCE"

    install_seed("reviewer", registry=registry, overwrite=True)
    reset = registry.get_agent_by_name("reviewer")
    assert reset is not None
    assert f"class:{REACTIVE}" in reset.tags


def test_a_row_ahead_of_the_packaged_starter_is_left_alone(scratch_seeds, tmp_path) -> None:
    """F6: an older build must not flip a newer text back (no ping-pong).

    Build the shape directly: publish a newer revision, install it, then roll
    the PACKAGE back to an older published text. The row now holds a revision
    NEWER than the package ships, and sync must report it without downgrading
    - the direction the version strings cannot express.
    """

    registry = AgentRegistry(tmp_path / "config")
    original = load_seed("reviewer")
    assert original is not None
    original_body = original.instructions
    original_version = load_seed_version("reviewer")
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    row = _install(registry)
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"

    move_seed(scratch_seeds, "reviewer", version=original_version, body=original_body)
    before = registry.get_agent_system_prompt(row.id)

    (verdict,) = sync_installed_seeds(registry)

    assert verdict.verdict == "up-to-date"
    assert "newer than this build ships" in verdict.detail
    assert registry.get_agent_system_prompt(row.id) == before


def test_the_legacy_stamp_formula_matches_the_shipped_era() -> None:
    """``_legacy_fingerprint`` is the OLD formula, digit for digit.

    The re-proof above rides on this agreeing with what v0.63.5-v0.64.8
    actually wrote; a silent drift here would re-break every legacy stamp.
    """

    from local_operator.agent_profiles import _legacy_fingerprint

    seed = load_seed("reviewer")
    assert seed is not None
    # Same five fields, same canonicalisation, computed independently.
    values = [
        (seed.instructions or "").strip(),
        (seed.when_to_use or seed.description or "").strip(),
        tuple(seed.tools) if seed.tools else None,
        seed.effort or None,
        bool(seed.may_delegate),
    ]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert _legacy_fingerprint(seed) == expected


# -- the startup pass, straight through its public entry point -----------------


def _drift_reviewer(seeds_dir: Path, config_dir: Path, **publish: Any) -> AgentRegistry:
    """An installed, PUBLISHED-behind reviewer row over one config dir.

    The shared set-up of every startup cell: install the starter, publish a
    move (seed file + ledger entry), and hand back the registry. ``publish``
    flows to :func:`publish_seed` (``version``/``body``/``tools``), so a cell
    varies only the delta under test.
    """

    registry = AgentRegistry(config_dir)
    assert _install(registry) is not None
    publish.setdefault("version", "2.0.0")
    publish.setdefault("body", "REVIEWER v2 GUIDANCE")
    publish_seed(seeds_dir, "reviewer", **publish)
    return registry


def _uv_tool_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend this interpreter is a real updatable install.

    The worktree venv is EDITABLE, and the pass is report-only there BY
    DESIGN (F6: worktree venvs share the operator's real config dir) - so
    every cell that expects a WRITE must patch the install kind through the
    same lazy seam a QA wrapper uses.
    """

    from local_operator.update import InstallKind

    monkeypatch.setattr("local_operator.update.install_kind", lambda **kw: InstallKind.UV_TOOL)


def test_the_startup_pass_applies_a_published_update_once_and_notices_once(
    scratch_seeds, tmp_path, monkeypatch, caplog
) -> None:
    """The #2060 headline: upgrade lop, and an untouched starter moves by itself.

    One pass, three properties: the update lands through the narrow writer
    (F4 - everything the seed does not own is preserved), the user gets ONE
    notice, and the SECOND launch is silent: the row is current and the
    notice's de-dup key is recorded, so neither a write nor a repeat happens.
    """

    from local_operator.agent_profiles import startup_seed_update_pass

    config_dir = tmp_path / "config"
    registry = AgentRegistry(config_dir)
    row = _install(registry, "aida")
    registry.update_agent(
        row.id,
        _edit_fields(label="My Aida", model="gpt-test-pin", tags=[*row.tags, "favourite"]),
    )
    refreshed = registry.get_agent_by_name("aida")
    assert refreshed is not None
    legacy = _legacy_fingerprint_of(registry, refreshed)
    registry.update_agent(
        refreshed.id,
        _edit_fields(
            tags=[t for t in refreshed.tags if not t.startswith(SEED_SHA256_PREFIX)]
            + [f"{SEED_SHA256_PREFIX}{legacy}"]
        ),
    )
    publish_seed(scratch_seeds, "aida", version="9.9.9", body="AIDA v9.9.9 GUIDANCE")
    _uv_tool_install(monkeypatch)

    with caplog.at_level(logging.INFO, logger="local_operator.agent_profiles"):
        outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ("aida",)
    assert len(outcome.announced) == 1
    assert "updated to the packaged starter" in outcome.announced[0]
    assert any("updated to the packaged starter" in record.message for record in caplog.records)

    after = AgentRegistry(config_dir).get_agent_by_name("aida")
    assert after is not None
    assert after.label == "My Aida"
    assert after.model == "gpt-test-pin"
    assert "favourite" in after.tags
    assert "class:proactive" in after.tags
    assert (
        AgentRegistry(config_dir).get_agent_system_prompt(after.id).strip()
        == "AIDA v9.9.9 GUIDANCE"
    )

    state = json.loads((config_dir / ".seed-notices.json").read_text(encoding="utf-8"))
    assert len(state["announced"]) == 1
    # The CLI surface logs; it does not queue for a TUI that is not there.
    assert state["pending"] == []

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="local_operator.agent_profiles"):
        second = startup_seed_update_pass(config_dir)
    assert second.applied == ()
    assert second.announced == ()
    assert not any("updated to the packaged starter" in record.message for record in caplog.records)


def test_the_startup_pass_holds_a_capability_change_and_says_why(
    scratch_seeds, tmp_path, monkeypatch
) -> None:
    """A delta touching ``tools:`` is NEVER applied unattended (F5).

    Two of 33 historical updates changed a capability field, and a widened
    allowlist is a fail-open boundary a person should see before it moves. The
    row is held with a notice naming what changes; the agent store stays
    byte-identical.
    """

    from local_operator.agent_profiles import startup_seed_update_pass

    config_dir = tmp_path / "config"
    _drift_reviewer(scratch_seeds, config_dir, tools="bash, read")
    _uv_tool_install(monkeypatch)
    before = _tree_bytes(config_dir / "agents")

    outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ()
    assert outcome.held == ("reviewer",)
    assert len(outcome.announced) == 1
    assert "tool access" in outcome.announced[0]
    assert _tree_bytes(config_dir / "agents") == before


def test_the_startup_pass_reports_only_when_the_setting_is_off(
    scratch_seeds, tmp_path, monkeypatch
) -> None:
    """``agents.auto_update.seeds: false`` stops the write, not the report.

    Absent means the shipped default (on); an explicit off turns every
    eligible row into an "available" notice the person can act on - the
    setting says "only tell me", so the pass must still TELL.
    """

    import yaml

    from local_operator.agent_profiles import startup_seed_update_pass

    config_dir = tmp_path / "config"
    _drift_reviewer(scratch_seeds, config_dir)
    (config_dir / "config.yml").write_text(
        yaml.safe_dump({"values": {"agents": {"auto_update": {"seeds": False}}}}),
        encoding="utf-8",
    )
    _uv_tool_install(monkeypatch)
    before = _tree_bytes(config_dir / "agents")

    outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ()
    assert outcome.available == ("reviewer",)
    assert len(outcome.announced) == 1
    assert "update to the packaged starter is available" in outcome.announced[0]
    assert _tree_bytes(config_dir / "agents") == before


def test_the_startup_pass_reports_only_on_an_editable_install(
    scratch_seeds, tmp_path, monkeypatch
) -> None:
    """A dev venv shares the operator's store and must never self-write (F6)."""

    from local_operator.agent_profiles import startup_seed_update_pass
    from local_operator.update import InstallKind

    config_dir = tmp_path / "config"
    _drift_reviewer(scratch_seeds, config_dir)
    monkeypatch.setattr("local_operator.update.install_kind", lambda **kw: InstallKind.EDITABLE)
    before = _tree_bytes(config_dir / "agents")

    outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ()
    assert outcome.available == ("reviewer",)
    assert len(outcome.announced) == 1
    assert _tree_bytes(config_dir / "agents") == before


def test_the_startup_pass_is_silent_for_an_edited_row(scratch_seeds, tmp_path, monkeypatch) -> None:
    """A row a person edited is never nagged about and never touched (Q1).

    The refusal to place an edited row is the whole direction-of-update
    argument: without proof of what the row was, no unattended write. Silence
    is the deliberate part - a dev build with drafts must not pop notices.
    """

    from local_operator.agent_profiles import startup_seed_update_pass

    config_dir = tmp_path / "config"
    registry = AgentRegistry(config_dir)
    row = _install(registry)
    registry.set_agent_system_prompt(row.id, "MY EDITED PROMPT")
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    _uv_tool_install(monkeypatch)

    outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ()
    assert outcome.announced == ()
    assert not (config_dir / ".seed-notices.json").exists()
    assert registry.get_agent_system_prompt(row.id) == "MY EDITED PROMPT"


def test_the_startup_pass_is_silent_for_a_row_ahead(scratch_seeds, tmp_path, monkeypatch) -> None:
    """A row holding a NEWER revision than the package ships stays put (F6)."""

    from local_operator.agent_profiles import startup_seed_update_pass

    config_dir = tmp_path / "config"
    registry = AgentRegistry(config_dir)
    original = load_seed("reviewer")
    assert original is not None
    original_body = original.instructions
    original_version = load_seed_version("reviewer")
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    row = _install(registry)
    move_seed(scratch_seeds, "reviewer", version=original_version, body=original_body)
    _uv_tool_install(monkeypatch)

    outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ()
    assert outcome.announced == ()
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"


def test_the_startup_pass_is_silent_with_no_readable_ledger(
    scratch_seeds, tmp_path, monkeypatch
) -> None:
    """Corrupt/missing ledger ends the pass silently - no positions, no proof."""

    from local_operator.agent_profiles import (
        SEED_REVISIONS_NAME,
        startup_seed_update_pass,
    )

    config_dir = tmp_path / "config"
    registry = AgentRegistry(config_dir)
    _install(registry)
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    (scratch_seeds / SEED_REVISIONS_NAME).write_bytes(b"\xff\xfe not json")
    _uv_tool_install(monkeypatch)
    before = _tree_bytes(config_dir / "agents")

    outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ()
    assert outcome.announced == ()
    assert _tree_bytes(config_dir / "agents") == before


def test_the_startup_pass_queues_pending_notices_for_the_tui_surface(
    scratch_seeds, tmp_path, monkeypatch
) -> None:
    """The TUI has no console at the seam's time, so the pass QUEUES there.

    ``surface="tui"`` writes the notice to ``.seed-notices.json`` ``pending``
    (announced also recorded) instead of logging; the TUI boot hook drains it.
    The cli surface is asserted elsewhere to leave ``pending`` empty.
    """

    from local_operator.agent_profiles import startup_seed_update_pass

    config_dir = tmp_path / "config"
    _drift_reviewer(scratch_seeds, config_dir)
    _uv_tool_install(monkeypatch)

    outcome = startup_seed_update_pass(config_dir, surface="tui")

    assert outcome.applied == ("reviewer",)
    state = json.loads((config_dir / ".seed-notices.json").read_text(encoding="utf-8"))
    assert state["pending"] == list(outcome.announced)
    assert len(state["announced"]) == 1


def test_the_startup_pass_leaves_a_storeless_machine_storeless(tmp_path) -> None:
    """No ``agents/`` store means the pass returns before constructing anything.

    ``AgentRegistry.__init__`` would mkdir the store this machine deliberately
    does not have; the storeless guard is what keeps a machine that never
    installed a starter from growing one because it ran ``lop --version``'s
    neighbours. Pinned at the pass AND at the seam (test_config.py).
    """

    from local_operator.agent_profiles import startup_seed_update_pass

    config_dir = tmp_path / "config"

    outcome = startup_seed_update_pass(config_dir)

    assert outcome.applied == ()
    assert outcome.announced == ()
    assert not config_dir.exists()


def test_the_startup_seam_skips_the_pass_for_agents_sync(
    scratch_seeds, tmp_path, monkeypatch
) -> None:
    """D3: the whole ``agents sync`` command skips the startup pass.

    ``lop agents sync --check`` promises "change nothing"; a startup
    auto-apply under the same invocation would break the promise AND change
    the state being checked. The control run right after proves the setup
    would otherwise have written - a skip test without one could pass on a
    broken fixture.
    """

    from local_operator.config_migrations import run_startup_migrations

    config_dir = tmp_path / "config"
    registry = AgentRegistry(config_dir)
    row = _install(registry)
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    _uv_tool_install(monkeypatch)
    before = _tree_bytes(config_dir / "agents")

    run_startup_migrations(config_dir, surface="cli", command="agents sync")

    assert _tree_bytes(config_dir / "agents") == before
    assert not (config_dir / ".seed-notices.json").exists()

    run_startup_migrations(config_dir, surface="cli")

    assert (
        AgentRegistry(config_dir).get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"
    )


def test_the_startup_pass_skips_silently_when_the_lock_is_held(
    scratch_seeds, tmp_path, monkeypatch
) -> None:
    """Several ``lop`` processes start at once; a held lock defers, silently."""

    from local_operator.agent_profiles import startup_seed_update_pass
    from local_operator.wakes.lock import WakeWriteLock

    config_dir = tmp_path / "config"
    registry = AgentRegistry(config_dir)
    _install(registry)
    publish_seed(scratch_seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    _uv_tool_install(monkeypatch)
    before = _tree_bytes(config_dir / "agents")

    holder = WakeWriteLock(config_dir, name=".seed-sync.lock", timeout_s=0.1)
    holder.acquire()
    try:
        outcome = startup_seed_update_pass(config_dir)
    finally:
        holder.release()

    assert outcome.applied == ()
    assert outcome.announced == ()
    assert outcome.skipped == ("reviewer",)
    assert _tree_bytes(config_dir / "agents") == before
