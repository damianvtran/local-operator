"""Agent role profiles: seeds, registry resolution, and the tool surface.

The behaviour under test is the CONTRACT of a role, never the prose of a
particular seed: the seed bodies are editable operator-facing files, so a test
that pinned their wording would turn every improvement to the guidance into a
test failure.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.agent_profiles import (
    MAX_INSTRUCTIONS_CHARS,
    READ_ONLY_NETWORK_TOOLS,
    READ_ONLY_TOOLS,
    AgentProfile,
    filter_tools,
    install_seed,
    list_seeds,
    load_seed,
    resolve_profile,
    seed_tags,
)
from local_operator.agents import AgentRegistry


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
    assert seed_tags(AgentProfile(name="plain")) == ("role",)
    tags = seed_tags(AgentProfile(name="x", tools=("read",), effort="lo", may_delegate=True))
    assert set(tags) == {"role", "tools:read", "effort:lo", "delegate:yes"}


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
