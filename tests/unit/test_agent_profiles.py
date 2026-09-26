"""Agent role profiles: seeds, registry resolution, and the tool surface.

The behaviour under test is the CONTRACT of a role, never the prose of a
particular seed: the seed bodies are editable operator-facing files, so a test
that pinned their wording would turn every improvement to the guidance into a
test failure.
"""

from __future__ import annotations

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
    AgentProfile,
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
    assert verdict.installed_version == "1.0.0"
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
    assert "force" in verdict.detail
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
