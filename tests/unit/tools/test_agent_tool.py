"""The ``agent`` tool: role discovery, installation, and authoring."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.agent_profiles import load_seed, resolve_profile
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.harness.types import ToolContext
from local_operator.tools.agent_tool import build_agent_tool, execute_agent


@pytest.fixture()
def registry(tmp_path) -> AgentRegistry:
    return AgentRegistry(tmp_path)


@pytest.fixture()
def context(registry: AgentRegistry) -> ToolContext:
    return _context(agent_registry=registry)


def _context(**kwargs: Any) -> ToolContext:
    return ToolContext(cwd=".", **kwargs)


def _edit_fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with every field spelled out (it is validated in
    strict mode), overridden by the few a test cares about."""
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


async def call(context: ToolContext, **args) -> str:
    result = await execute_agent("tc", args, None, None, context)
    return result.text


def test_the_tool_is_not_advertised_without_a_registry() -> None:
    """createIf: a tool that could show a role but never keep one is a worse
    surface than no tool."""
    assert build_agent_tool(_context()) is None
    assert build_agent_tool(_context(agent_registry=object())) is not None


@pytest.mark.asyncio
async def test_list_offers_starters_before_anything_is_installed(context) -> None:
    body = await call(context, op="list")
    assert "installable starters" in body
    assert "reviewer" in body


@pytest.mark.asyncio
async def test_list_separates_registered_roles_from_starters(context, registry) -> None:
    await call(context, op="install", name="reviewer")
    body = await call(context, op="list")
    registered, _, starters = body.partition("installable starters")
    assert "reviewer" in registered
    assert "reviewer" not in starters, "an installed role must not still be offered"
    assert "architect" in starters


@pytest.mark.asyncio
async def test_list_and_show_include_specialists_with_instructions(context) -> None:
    await call(
        context,
        op="create",
        kind="specialist",
        name="dashboard-release",
        description="Release the user dashboard safely",
        instructions="Follow the dashboard SDLC and release checklist.",
    )

    listing = await call(context, op="list")
    shown = await call(context, op="show", name="dashboard-release")

    assert "registered specialists" in listing
    assert "dashboard-release" in listing
    assert "Follow the dashboard SDLC" in shown
    assert "launch with --agent dashboard-release" in shown


@pytest.mark.asyncio
async def test_list_ignores_agents_that_are_not_roles(context, registry) -> None:
    """A registry also holds ordinary conversational agents; listing those as
    delegation targets would be noise at best and a privacy leak at worst."""
    registry.create_agent(_edit_fields(name="my-private-notes", description="personal"))
    body = await call(context, op="list")
    assert "my-private-notes" not in body


@pytest.mark.asyncio
async def test_search_finds_the_right_role_from_a_task_description(context) -> None:
    """The routing case: the delegator describes the work, not the role name."""
    body = await call(context, op="search", query="check this pull request diff for bugs")
    assert "reviewer" in body


@pytest.mark.asyncio
async def test_search_finds_a_role_authored_a_moment_ago(context) -> None:
    """No rebuild step: a role created in this session is routable now."""
    await call(
        context,
        op="create",
        name="sec-auditor",
        description="Security audit of authentication and tenant isolation code paths",
        instructions="Audit for authz gaps and tenant leakage. Cite file:line.",
    )
    body = await call(context, op="search", query="look for tenant isolation problems in auth")
    assert "sec-auditor" in body


@pytest.mark.asyncio
async def test_show_returns_the_full_instructions(context) -> None:
    body = await call(context, op="show", name="reviewer")
    assert "instructions:" in body
    assert "BLOCKER" in body
    assert "not installed" in body


@pytest.mark.asyncio
async def test_install_makes_the_role_launchable(context, registry) -> None:
    body = await call(context, op="install", name="reviewer")
    assert "installed role" in body
    assert resolve_profile("reviewer", registry=registry) is not None


@pytest.mark.asyncio
async def test_create_then_update_changes_what_the_role_is_told(context, registry) -> None:
    await call(
        context,
        op="create",
        name="triager",
        description="Triage incoming bug reports and route them",
        instructions="ORIGINAL",
        tools=["read", "grep"],
        effort="lo",
    )
    profile = resolve_profile("triager", registry=registry)
    assert profile is not None
    assert profile.instructions == "ORIGINAL"
    assert profile.tools == ("read", "grep")
    assert profile.effort == "lo"

    await call(context, op="update", name="triager", instructions="REVISED")
    revised = resolve_profile("triager", registry=registry)
    assert revised is not None and revised.instructions == "REVISED"


@pytest.mark.asyncio
async def test_create_refuses_to_clobber_an_existing_role(context) -> None:
    await call(context, op="create", name="triager", description="d", instructions="ORIGINAL")
    body = await call(context, op="create", name="triager", description="d", instructions="NEW")
    assert "already exists" in body


@pytest.mark.asyncio
async def test_update_refuses_an_unknown_role(context) -> None:
    assert "no registered profile" in await call(
        context, op="update", name="ghost", instructions="x"
    )


@pytest.mark.asyncio
async def test_create_specialist_is_not_a_role(context, registry) -> None:
    """A User Dashboard Agent is a reusable specialist, not a delegation role."""
    body = await call(
        context,
        op="create",
        name="user-dashboard",
        kind="specialist",
        description="Knows user-dashboard release practices",
        instructions="Follow the user-dashboard SDLC. Never skip the design review.",
    )
    assert "created specialist" in body
    agent = registry.get_agent_by_name("user-dashboard")
    assert agent is not None
    assert "role" not in (agent.tags or [])
    assert resolve_profile("user-dashboard", registry=registry) is None
    prompt = registry.get_agent_system_prompt(agent.id)
    assert "user-dashboard SDLC" in prompt


@pytest.mark.asyncio
async def test_create_needs_instructions(context) -> None:
    assert "needs 'instructions'" in await call(context, op="create", name="empty", description="d")


@pytest.mark.asyncio
async def test_oversized_instructions_are_refused(context) -> None:
    """They ride in front of every run of the role, so the bound is the point."""
    body = await call(context, op="create", name="huge", description="d", instructions="x" * 20_000)
    assert "exceed" in body


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["show", "install", "create", "update"])
async def test_ops_that_need_a_name_say_so(context, op: str) -> None:
    assert "needs 'name'" in await call(context, op=op)


@pytest.mark.asyncio
async def test_search_needs_a_query(context) -> None:
    assert "needs 'query'" in await call(context, op="search")


@pytest.mark.asyncio
async def test_installing_an_unknown_starter_lists_the_real_ones(context) -> None:
    body = await call(context, op="install", name="nope")
    assert "no packaged starter" in body and "reviewer" in body


# ---------------------------------------------------------------------------
# Round-1 review regressions. The originals all passed a green suite because
# they only ever exercised a freshly INSTALLED seed; these exercise the state
# after an edit, and the collision with a same-named non-role agent.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_keeps_the_allowlist_it_was_not_asked_to_change(context, registry) -> None:
    """C1: refining a role's wording used to strip its `tools:` tag, handing a
    reviewer `edit`/`write` — failing OPEN while reporting success."""
    await call(context, op="install", name="reviewer")
    before = resolve_profile("reviewer", registry=registry)
    assert before is not None and before.tools

    await call(context, op="update", name="reviewer", instructions="Read the diff first.")

    after = resolve_profile("reviewer", registry=registry)
    assert after is not None
    assert after.tools == before.tools, "the allowlist is a capability boundary"
    assert "Read the diff first." in after.instructions


@pytest.mark.asyncio
async def test_update_keeps_the_effort_tier_and_delegation_flag(context, registry) -> None:
    """C1, the same bug on the other two role fields. Manager ships with
    may_delegate; its tier is set explicitly here because the packaged seeds
    deliberately pin NO effort — a child inherits the session's model unless
    the operator picks a tier — so the seed itself cannot supply one."""
    await call(context, op="install", name="manager")
    await call(context, op="update", name="manager", effort="lo")
    before = resolve_profile("manager", registry=registry)
    assert before is not None and before.effort == "lo" and before.may_delegate

    await call(context, op="update", name="manager", description="new routing text")

    after = resolve_profile("manager", registry=registry)
    assert after is not None
    assert after.effort == before.effort
    assert after.may_delegate == before.may_delegate
    assert after.tools == before.tools


@pytest.mark.asyncio
async def test_an_explicit_empty_tools_list_still_clears_the_allowlist(context, registry) -> None:
    """Preserving an omitted field must not make the field unclearable:
    NAMING it with an empty value is how you widen a role deliberately."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", tools=[])
    profile = resolve_profile("reviewer", registry=registry)
    assert profile is not None and profile.tools is None


@pytest.mark.asyncio
async def test_install_refuses_a_name_owned_by_a_non_role(context, registry) -> None:
    """C4: it used to report a successful install while writing nothing, which
    is the worst possible answer to someone recovering from a broken role."""
    agent = registry.create_agent(_edit_fields(name="reviewer", description="my chat agent"))
    registry.set_agent_system_prompt(agent.id, "Be agreeable.")

    body = await call(context, op="install", name="reviewer")

    assert "not a role" in body and "nothing was installed" in body
    assert registry.get_agent_system_prompt(agent.id) == "Be agreeable.", "must not overwrite"


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["create", "update"])
async def test_authoring_refuses_a_name_owned_by_a_non_role(context, registry, op: str) -> None:
    """An ordinary agent must not silently become a role, and on the update
    path converting one would be the same fail-open hijack."""
    registry.create_agent(_edit_fields(name="reviewer", description="my chat agent"))
    body = await call(context, op=op, name="reviewer", description="d", instructions="x")
    assert "not a role" in body


@pytest.mark.asyncio
async def test_a_role_can_be_authored_as_a_delegating_one(context, registry) -> None:
    """C14: `may_delegate` was preserved across updates but unreachable — no
    role authored through this tool could ever coordinate, and an installed
    manager's flag could not be revoked."""
    await call(
        context,
        op="create",
        name="lead",
        description="coordinates multi-part work",
        instructions="Coordinate.",
        delegate=True,
    )

    def delegates() -> bool:
        profile = resolve_profile("lead", registry=registry)
        assert profile is not None
        return profile.may_delegate

    assert delegates() is True

    await call(context, op="update", name="lead", instructions="Coordinate well.")
    assert delegates() is True, "preserved across an unrelated update"

    await call(context, op="update", name="lead", delegate=False)
    assert delegates() is False, "revocable"


@pytest.mark.asyncio
async def test_the_tool_returns_an_error_result_instead_of_raising(context, registry) -> None:
    """C13: `agent` was the only executor in the tree without `@_guard`, and
    the harness contract is that tools never throw into the loop."""

    def explode(*args, **kwargs):
        raise PermissionError("agents dir is read-only")

    registry.create_agent = explode  # type: ignore[method-assign]
    result = await execute_agent(
        "tc",
        {"op": "create", "name": "x", "description": "d", "instructions": "i"},
        None,
        None,
        context,
    )
    assert result.is_error
    assert "PermissionError" in result.text or "read-only" in result.text


# ---------------------------------------------------------------------------
# Design review round-1 regressions (copy surface).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unrestricted_role_is_not_shown_as_the_smallest_one(context) -> None:
    """D2: the badge was emitted only when an allowlist existed, so the roles
    with FULL write access rendered as a bare `[starter]` while read-only
    `scout` showed `[5 tools]` — inverting the one attribute the guide calls a
    capability boundary."""
    body = await call(context, op="list")
    rows = {line.split()[1]: line for line in body.splitlines() if line.startswith("- ")}
    assert "all tools" in rows["coder"], rows["coder"]
    assert "all tools" in rows["designer"], rows["designer"]
    # The COUNT tracks the seed's allowlist and is asserted from it rather than
    # hard-coded: the read-only surface gained the network tools, and a literal
    # here would have to be re-guessed on every such change while testing
    # nothing about the badge.
    scout = load_seed("scout")
    assert scout is not None and scout.tools
    assert f"{len(scout.tools)} tools" in rows["scout"], rows["scout"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("find where this function is defined", "scout"),
        ("write the code for this ticket", "coder"),
        ("make the button look nicer", "designer"),
        ("audit this diff for problems", "reviewer"),
        ("decide between two architectures", "architect"),
    ],
)
async def test_search_answers_the_phrasings_the_seeds_advertise(
    context, query: str, expected: str
) -> None:
    """D1: the shared index threshold (0.19) is calibrated for full skill
    bodies, so one-sentence role descriptions scored 0.118-0.177 and were cut —
    `scout.md` promises "locating code" while that query returned nothing. The
    ranking is now surfaced best-first rather than gated on an absolute score.
    """
    body = await call(context, op="search", query=query)
    rows = [line for line in body.splitlines() if line.startswith("- ")]
    assert rows, f"{query!r} matched no role"
    assert rows[0].split()[1] == expected, f"{query!r} -> {rows[0]}"


@pytest.mark.asyncio
async def test_show_on_an_uninstalled_starter_names_the_next_command(context) -> None:
    """D3: it stated "not installed" and stopped, at the moment the reader has
    just decided they want the role."""
    body = await call(context, op="show", name="coder")
    assert "task(agent='coder')" in body
    assert "op='install'" in body


@pytest.mark.asyncio
async def test_a_role_with_no_description_says_it_is_undiscoverable(context) -> None:
    """D6: an empty description rendered as a dangling `- name: `, hiding that
    `search` matches on exactly that text."""
    await call(context, op="create", name="nodesc", instructions="does a thing")
    row = next(line for line in (await call(context, op="list")).splitlines() if "nodesc" in line)
    assert "not searchable" in row


@pytest.mark.asyncio
async def test_an_invalid_effort_tier_is_refused(context) -> None:
    """D5: it was accepted silently and then rendered in the badge slot as
    though it were a real tier, teaching the reader that any token belongs
    there."""
    body = await call(
        context, op="create", name="e1", description="d", instructions="i", effort="ludicrous"
    )
    assert "invalid arguments" in body
    assert "effort" in body


@pytest.mark.asyncio
async def test_an_overlong_row_is_marked_as_truncated(context, registry) -> None:
    """D4: a bare slice ends mid-word, so a reader cannot tell an author's
    fragment from text the tool dropped."""
    await call(context, op="create", name="wordy", description="x " * 400, instructions="i")
    row = next(line for line in (await call(context, op="list")).splitlines() if "wordy" in line)
    assert row.endswith("…")


# ---------------------------------------------------------------------------
# Design review round-2 regressions.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("check the UI looks right", "designer"),
        ("find where this function is defined", "scout"),
        ("write the code for this ticket", "coder"),
        ("make the button look nicer", "designer"),
        ("fix this bug", "coder"),
    ],
)
async def test_search_still_works_after_the_roles_are_installed(
    context, query: str, expected: str
) -> None:
    """D10: `install_seed` persisted `description` while the enriched trigger
    text lived in `when_to_use`, so installing a role — the flow the list
    header advertises — discarded exactly the text `search` matches on. Four of
    five of the round-1 queries then resolved to the WRONG role, and because
    the absolute threshold had been removed the failure was confident rather
    than visible ("check the UI looks right" -> manager).

    The round-1 tests only exercised packaged starters, which is why this
    escaped; this one installs first.
    """
    for name in ("reviewer", "coder", "designer", "scout", "architect", "manager"):
        await call(context, op="install", name=name)

    body = await call(context, op="search", query=query)
    rows = [line for line in body.splitlines() if line.startswith("- ")]
    assert rows, f"{query!r} matched no role after install"
    assert rows[0].split()[1] == expected, f"{query!r} -> {rows[0]}"


@pytest.mark.asyncio
async def test_the_listing_stays_scannable(context) -> None:
    """D11: enriching the trigger text for retrieval pushed rows to 188-246
    chars, so six roles rendered as 22 physical lines at 80 columns with no
    indent marking where one ended. Display now prefers the short description
    and the row cap bounds the rest."""
    body = await call(context, op="list")
    rows = [line for line in body.splitlines() if line.startswith("- ")]
    assert rows
    assert all(len(row) <= 160 for row in rows), sorted(len(row) for row in rows)
    physical = sum(-(-len(row) // 80) for row in rows)
    assert physical <= 14, f"{physical} physical lines at 80 columns"
    header = next(line for line in body.splitlines() if "installable starters" in line)
    assert len(header) <= 80, len(header)


@pytest.mark.asyncio
async def test_a_weak_search_result_is_worded_as_a_nearest_neighbour(context) -> None:
    """D12: with no absolute cut, an unrelated query still returns three roles,
    and nothing in a bare list distinguished a 0.65 hit from 0.03 noise."""
    strong = await call(context, op="search", query="review a merge request for defects")
    assert strong.startswith("closest roles (best first):")

    weak = await call(context, op="search", query="order me a pizza")
    assert weak.startswith("nothing scored strongly")
    assert "- " in weak, "the shortlist is still shown, just not as a recommendation"


@pytest.mark.asyncio
async def test_search_only_claims_best_first_when_it_ranked(context, monkeypatch) -> None:
    """C20: `_ranked_names`' fallback returns [], which leaves the rows in
    `select`'s NAME order — so the header claimed "best first" over an
    alphabetical list, the exact hazard the helper's docstring names. The
    header is honest exactly when the helper works and wrong when it does not.
    """
    import local_operator.skills.index as index_module

    healthy = await call(context, op="search", query="review a merge request for defects")
    assert healthy.startswith("closest roles (best first):")

    def boom(*args: Any, **kwargs: Any):
        raise TypeError("signature drifted")

    monkeypatch.setattr(index_module, "_hybrid_scores", boom)
    degraded = await call(context, op="search", query="review a merge request for defects")
    assert degraded.startswith("closest roles:")
    assert "best first" not in degraded
    assert "- " in degraded, "the shortlist is still returned, just unordered"


@pytest.mark.asyncio
async def test_reinstalling_an_edited_role_reports_the_no_op(context, registry) -> None:
    """D14: `install_seed` is deliberately idempotent, but the message did not
    say so — and `install` is the only verb here that sounds like "put the
    shipped one back". An operator recovering from a role they broke was told
    `installed role 'reviewer'` while their own edited prompt remained what the
    next delegation would run. Same failure shape C4 fixed on the non-role
    branch.
    """
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="ONLY CHECK THE MIGRATIONS.")

    body = await call(context, op="install", name="reviewer")

    assert "already installed" in body and "left as-is" in body
    profile = resolve_profile("reviewer", registry=registry)
    assert profile is not None
    assert "MIGRATIONS" in profile.instructions, "the edit must survive, and it does"


@pytest.mark.asyncio
async def test_a_single_tool_role_is_not_labelled_one_tools(context) -> None:
    """D15: the badge was built unconditionally as `{n} tools`."""
    await call(context, op="create", name="one", description="d", instructions="i", tools=["read"])
    row = next(line for line in (await call(context, op="list")).splitlines() if "one" in line)
    assert "[1 tool]" in row and "1 tools" not in row


@pytest.mark.asyncio
async def test_show_prints_the_packaged_text_when_an_installed_role_diverged(
    context, registry
) -> None:
    """#141, option 2: once a packaged role is installed and edited, the shipped
    guidance is unreachable — `show` renders the edit and `update` asks the
    reader to retype text they can no longer read. The reader's actual complaint
    is that they cannot see what they lost."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="ONLY CHECK THE MIGRATIONS.")

    body = await call(context, op="show", name="reviewer")

    assert "ONLY CHECK THE MIGRATIONS." in body, "their own text is still what runs"
    assert "differs from the packaged starter" in body
    assert "You are an INDEPENDENT reviewer" in body, "the packaged text is readable again"
    assert "op='reset' name='reviewer'" in body


@pytest.mark.asyncio
async def test_show_stays_quiet_when_an_installed_role_still_matches(context) -> None:
    """The divergence block must be a signal, not decoration. `install_seed`
    rewrites the registry description to the seed's routing text, so comparing
    anything but the instruction body would flag every installed role."""
    await call(context, op="install", name="reviewer")

    body = await call(context, op="show", name="reviewer")

    assert "differs from the packaged starter" not in body
    assert "op='reset'" not in body


@pytest.mark.asyncio
async def test_show_on_a_user_authored_role_never_offers_a_reset(context) -> None:
    """A role with no packaged counterpart has nothing to be compared against,
    let alone restored to."""
    await call(context, op="create", name="mine", description="d", instructions="my guidance")

    body = await call(context, op="show", name="mine")

    assert "differs from the packaged starter" not in body and "op='reset'" not in body


@pytest.mark.asyncio
async def test_reset_restores_the_packaged_role_and_reports_what_it_replaced(
    context, registry
) -> None:
    """#141, option 1: the verb a reader looks for by name. It must also echo
    the replaced text — a reset that silently discards operator guidance is a
    data loss with a friendly message on it."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="ONLY CHECK THE MIGRATIONS.")

    body = await call(context, op="reset", name="reviewer")

    assert "reset role 'reviewer'" in body
    assert "ONLY CHECK THE MIGRATIONS." in body, "the overwrite stays recoverable by copy-paste"
    profile = resolve_profile("reviewer", registry=registry)
    packaged = load_seed("reviewer")
    assert profile is not None and packaged is not None
    assert "MIGRATIONS" not in profile.instructions
    assert profile.instructions.strip() == packaged.instructions.strip()


@pytest.mark.asyncio
async def test_reset_keeps_the_role_launchable_with_its_packaged_tool_surface(
    context, registry
) -> None:
    """A restored role must be the packaged role in every respect, not just its
    prose: the tool allowlist is a capability boundary, so a reset that dropped
    it would hand a reviewer the full write inventory."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", tools=["read"], instructions="broken")

    await call(context, op="reset", name="reviewer")

    profile = resolve_profile("reviewer", registry=registry)
    packaged = load_seed("reviewer")
    assert profile is not None and packaged is not None
    assert profile.tools == packaged.tools
    assert profile.may_delegate == packaged.may_delegate


@pytest.mark.asyncio
async def test_reset_on_an_unedited_role_reports_the_no_op(context) -> None:
    """Reporting a restore for a no-op is the exact misreport that `install`'s
    message was fixed for (D14); the answer to "did it change?" has to be no."""
    await call(context, op="install", name="reviewer")

    body = await call(context, op="reset", name="reviewer")

    assert "already matches the packaged version" in body and "nothing was changed" in body


@pytest.mark.asyncio
async def test_reset_refuses_a_user_authored_role_rather_than_deleting_it(
    context, registry
) -> None:
    """A role nobody packaged has nothing to be restored TO, so a reset could
    only mean deleting it — never what the word means here."""
    await call(context, op="create", name="mine", description="d", instructions="my guidance")

    body = await call(context, op="reset", name="mine")

    assert "nothing to reset it to" in body and "nothing was changed" in body
    assert "op='update'" in body, "it must name the command that does work"
    profile = resolve_profile("mine", registry=registry)
    assert profile is not None and profile.instructions == "my guidance"


@pytest.mark.asyncio
async def test_reset_on_an_unknown_name_lists_the_real_starters(context) -> None:
    body = await call(context, op="reset", name="nope")
    assert "no packaged starter named 'nope'" in body
    assert "reviewer" in body and "architect" in body


@pytest.mark.asyncio
async def test_reset_needs_a_name(context) -> None:
    assert "needs 'name'" in await call(context, op="reset")


@pytest.mark.asyncio
async def test_reset_on_a_never_installed_starter_installs_it_and_says_so(context) -> None:
    """The end state the caller asked for is the one they get, but the report
    must not claim it replaced guidance that never existed."""
    body = await call(context, op="reset", name="scout")

    assert "was not installed" in body and "installed the packaged version" in body
    assert "It replaced" not in body


@pytest.mark.asyncio
async def test_reset_refuses_a_name_owned_by_a_non_role(context, registry) -> None:
    """`install_seed(overwrite=True)` skips its own non-role guard by design (the
    kwarg means "the caller has decided"), so this check is load-bearing: without
    it a reset would rewrite an ordinary chat agent's prompt with role guidance."""
    agent = registry.create_agent(_edit_fields(name="reviewer", description="my chat agent"))
    registry.set_agent_system_prompt(agent.id, "Be agreeable.")

    body = await call(context, op="reset", name="reviewer")

    assert "not a role" in body and "nothing was reset" in body
    assert registry.get_agent_system_prompt(agent.id) == "Be agreeable.", "must not overwrite"


@pytest.mark.asyncio
async def test_reinstall_names_reset_as_the_way_to_get_the_packaged_text_back(context) -> None:
    """#141: the pre-fix message offered `show` (which rendered their own broken
    text) and `update` (which asked them to retype what they could not read), so
    both offered commands dead-ended."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="ONLY CHECK THE MIGRATIONS.")

    body = await call(context, op="install", name="reviewer")

    assert "left as-is" in body, "install stays idempotent; that is what protects the edits"
    assert "op='reset' name='reviewer'" in body


@pytest.mark.parametrize(
    "field,edit",
    [
        ("tools", {"tools": ["read", "edit", "write"]}),
        ("effort", {"effort": "lo"}),
        ("delegate", {"delegate": True}),
    ],
)
@pytest.mark.asyncio
async def test_reset_restores_a_role_edited_only_in_a_non_prose_field(
    context, registry, field: str, edit: dict[str, Any]
) -> None:
    """F1/U1: the no-op short-circuit compared INSTRUCTIONS ONLY, so a role whose
    allowlist had been widened but whose prose was untouched never reached the
    overwrite. It answered "nothing was changed" and kept the widened surface —
    the exact fail-open the restore exists to close, while reporting success.

    Parametrized across every non-prose field the seed writes because the bug was
    not specific to `tools`; a check that only learned about `tools` would leave
    `effort` and `delegate` in the same hole.
    """
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", **edit)

    shown = await call(context, op="show", name="reviewer")
    body = await call(context, op="reset", name="reviewer")

    assert "differs from the packaged starter" in shown, "show must surface it too"
    assert field in shown
    assert "nothing was changed" not in body
    assert f"{field} replaced" in body
    profile = resolve_profile("reviewer", registry=registry)
    packaged = load_seed("reviewer")
    assert profile is not None and packaged is not None
    assert profile.tools == packaged.tools
    assert (profile.effort or None) == (packaged.effort or None)
    assert profile.may_delegate == packaged.may_delegate


@pytest.mark.asyncio
async def test_divergence_is_not_escapable_by_editing_one_character_of_prose(
    context,
) -> None:
    """F1/U1: with two independent comparisons the verdict on a widened allowlist
    flipped depending on whether the prose also happened to differ. `show` and
    `reset` now share one predicate, so the answer cannot depend on that."""
    packaged = load_seed("reviewer")
    assert packaged is not None

    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", tools=["read"])
    tools_only = "differs from the packaged starter" in await call(
        context, op="show", name="reviewer"
    )

    await call(context, op="update", name="reviewer", instructions=packaged.instructions + " extra")
    tools_and_prose = "differs from the packaged starter" in await call(
        context, op="show", name="reviewer"
    )

    assert tools_only is tools_and_prose is True


@pytest.mark.asyncio
async def test_reset_refuses_a_self_authored_role_under_a_packaged_name(context, registry) -> None:
    """F2: the guard was `load_seed(name) is None`, so a role the operator wrote
    themselves under a starter's name (`scout`) took the restore path and was
    destroyed. The refusal must key on PROVENANCE, not on name collision."""
    await call(context, op="create", name="scout", description="mine", instructions="MY OWN WORK")

    shown = await call(context, op="show", name="scout")
    body = await call(context, op="reset", name="scout")

    # `show` DOES report the difference (U7: reading is not destructive, and
    # going quiet here was the #141 dead end), but it must never offer `reset`
    # on a role that reset would refuse.
    assert "differs from the packaged starter" in shown
    assert "op='reset'" not in shown.split("launch with")[1]
    assert "no record of being installed" in body and "nothing was changed" in body
    assert "op='update' name='scout'" in body
    profile = resolve_profile("scout", registry=registry)
    assert profile is not None and profile.instructions == "MY OWN WORK"


@pytest.mark.asyncio
async def test_an_edit_does_not_strip_a_role_of_its_provenance(context, registry) -> None:
    """The provenance marker is rebuilt by `_op_write`, which encodes a PROFILE's
    fields and knows nothing about origin. Losing the marker on edit would make
    `reset` refuse exactly the roles it exists for: an edited install."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="ONLY CHECK THE MIGRATIONS.")

    body = await call(context, op="reset", name="reviewer")

    assert "reset role 'reviewer'" in body, "an edited install must still be resettable"
    profile = resolve_profile("reviewer", registry=registry)
    assert profile is not None and "MIGRATIONS" not in profile.instructions


@pytest.mark.asyncio
async def test_reset_echoes_every_field_it_replaced_not_only_the_prose(context) -> None:
    """U2: "recoverable by copy-paste" held for instructions only. A role someone
    had deliberately NARROWED to `read` lost that restriction with no record,
    while the closing line pointed at op='update' for the thing that WAS echoed."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", tools=["read"])

    body = await call(context, op="reset", name="reviewer")

    assert "tools replaced" in body
    assert "your tools: read" in body, "the restriction they lose has to be written down"
    assert "op='update'" in body


@pytest.mark.asyncio
async def test_a_one_line_edit_renders_as_a_diff_not_two_walls_of_text(context) -> None:
    """U3: the block printed both ~32-line bodies whole and left the reader to
    spot a one-line difference by eye, billed to their context."""
    packaged = load_seed("reviewer")
    assert packaged is not None
    lines = packaged.instructions.splitlines()
    tweaked = "\n".join(lines[:5] + ["You may skip the tests if you are in a hurry."] + lines[5:])

    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions=tweaked)

    block = (await call(context, op="show", name="reviewer")).split(
        "differs from the packaged starter"
    )[1]

    assert "as a diff against yours" in block
    assert "+++ packaged" in block
    assert "You may skip the tests" in block, "the differing line is what the reader needs"
    assert len(block.splitlines()) < len(lines), "a diff that is longer than the body is no diff"


@pytest.mark.asyncio
async def test_a_wholly_rewritten_role_falls_back_to_the_full_body(context) -> None:
    """A diff of two unrelated texts is longer than the text itself, so the
    fallback prints the body — and must not still call it a diff."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="totally different short text")

    block = (await call(context, op="show", name="reviewer")).split(
        "differs from the packaged starter"
    )[1]

    assert "too different to diff usefully" in block
    assert "as a diff against yours" not in block
    assert "You are an INDEPENDENT reviewer" in block


@pytest.mark.asyncio
async def test_reset_on_an_unknown_name_that_is_a_registered_role_names_update(
    context,
) -> None:
    """U4: sending the reader to a bare starter list was a second dead end when
    the name they typed is a role of their own with no packaged counterpart."""
    await call(context, op="create", name="mine", description="d", instructions="my guidance")

    body = await call(context, op="reset", name="mine")

    assert "You do have a role named 'mine'" in body
    assert "op='update' name='mine'" in body


@pytest.mark.asyncio
async def test_reset_leaves_operator_owned_settings_alone(context, registry) -> None:
    """F3: a reset restores the packaged ROLE, not a factory-reset row. `model`,
    `hosting`, `security_prompt` and sampling are the operator's; the seed pins
    none of them and `update_agent` cannot clear a field through this path."""
    await call(context, op="install", name="reviewer")
    agent = registry.get_agent_by_name("reviewer")
    registry.update_agent(
        agent.id, _edit_fields(security_prompt="stay in scope", model="gpt-4o-mini")
    )
    await call(context, op="update", name="reviewer", instructions="broken")

    await call(context, op="reset", name="reviewer")

    after = registry.get_agent_by_name("reviewer")
    assert after.security_prompt == "stay in scope"
    assert after.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_both_result_bodies_spell_the_launch_line_the_same_way(context) -> None:
    """F5: `show` said "launch with" and `_op_reset` said "Launch with"."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="broken")

    assert "Launch with task(" not in await call(context, op="reset", name="reviewer")
    assert "launch with task(" in await call(context, op="show", name="reviewer")


def _legacy_role_row(registry: AgentRegistry, name: str, **overrides: Any):
    """A role row as a PRE-PROVENANCE release wrote it: role tags, no `seed:`.

    The population U7 was found on. Built by hand because no code path writes
    an unmarked row any more, and the upgrade case cannot be tested without one.
    """
    from local_operator.agent_profiles import AgentProfile, seed_tags

    seed = load_seed(name)
    assert seed is not None
    profile = AgentProfile(
        name=name,
        tools=overrides.get("tools", seed.tools),
        effort=seed.effort,
        may_delegate=seed.may_delegate,
    )
    agent = registry.create_agent(
        _edit_fields(
            name=name,
            description=seed.when_to_use or seed.description,
            tags=list(seed_tags(profile)),
            categories=["role"],
        )
    )
    registry.set_agent_system_prompt(agent.id, overrides.get("instructions", seed.instructions))
    return agent


@pytest.mark.asyncio
async def test_a_legacy_row_with_untouched_prose_can_still_be_reset(context, registry) -> None:
    """U7: keying reset on the `seed:` marker alone locked out every role
    installed by an earlier release — on the real registry, the rows whose
    allowlist had drifted, which are exactly the users reset exists for. A row
    whose prose is byte-identical to the seed holds nobody's writing, so it
    unlocks without weakening the guard for a row that does."""
    _legacy_role_row(context.agent_registry, "reviewer", tools=("read", "bash"))

    body = await call(context, op="reset", name="reviewer")

    assert "tools replaced" in body
    profile = resolve_profile("reviewer", registry=registry)
    packaged = load_seed("reviewer")
    assert profile is not None and packaged is not None
    assert profile.tools == packaged.tools


@pytest.mark.asyncio
async def test_a_legacy_row_with_rewritten_prose_is_refused_without_claiming_authorship(
    context, registry
) -> None:
    """U7: absence of a marker is not evidence a human wrote the role. A
    pre-marker install and a hand-authored role are indistinguishable BY
    CONSTRUCTION, so the refusal must state what is known (no install record)
    rather than assert history it cannot support."""
    _legacy_role_row(context.agent_registry, "manager", instructions="MY OWN REWRITE")

    body = await call(context, op="reset", name="manager")

    assert "no record of being installed" in body
    assert "authored here" not in body, "do not assert authorship the tool cannot know"
    assert "packaged instructions:" in body, "the packaged text must be copy-pasteable"
    assert "op='update' name='manager'" in body
    profile = resolve_profile("manager", registry=registry)
    assert profile is not None and "MY OWN REWRITE" in profile.instructions


@pytest.mark.asyncio
async def test_show_still_reads_a_role_that_reset_will_not_overwrite(context) -> None:
    """U7: gating the READ path on provenance left a pre-upgrade user at the
    exact #141 dead end. Reading is not destructive, so `show` keeps working and
    points at the op that does — never at `reset`, which would refuse."""
    _legacy_role_row(context.agent_registry, "manager", instructions="MY OWN REWRITE")

    body = await call(context, op="show", name="manager")

    assert "differs from the packaged starter" in body
    tail = body.split("launch with")[1]
    assert "op='update'" in tail
    assert "op='reset'" not in tail, "never offer a verb that will refuse"


@pytest.mark.asyncio
async def test_an_unrecorded_role_that_already_matches_reports_the_plain_no_op(context) -> None:
    """U8: a legacy row that already matches was told it "was authored here",
    which over-claimed on the one case where the true answer is simply "nothing
    to do". The byte-identical unlock answers this structurally rather than by
    wording: matching prose means the row is treated as installed, so it takes
    the ordinary no-op path and never reaches a refusal at all."""
    _legacy_role_row(context.agent_registry, "coder")

    body = await call(context, op="reset", name="coder")

    assert "already matches the packaged version" in body
    assert "no record of being installed" not in body
    assert "authored here" not in body


@pytest.mark.asyncio
async def test_a_self_authored_role_is_still_refused_after_the_legacy_unlock(
    context, registry
) -> None:
    """The unlock must not reopen F2: a role written from scratch under a
    starter's name has prose that does NOT match, so it stays protected."""
    await call(context, op="create", name="scout", description="mine", instructions="MY OWN WORK")

    body = await call(context, op="reset", name="scout")

    assert "no record of being installed" in body
    profile = resolve_profile("scout", registry=registry)
    assert profile is not None and profile.instructions == "MY OWN WORK"


@pytest.mark.asyncio
async def test_refusals_name_only_commands_that_exist(context, registry) -> None:
    """U7: the old text sent readers to a rename and a delete this tool does not
    have, and the guide's `install` workaround errored. Naming a command that
    does not exist is the dead-end failure #141 was filed about."""
    agent = registry.create_agent(_edit_fields(name="reviewer", description="my chat agent"))
    registry.set_agent_system_prompt(agent.id, "Be agreeable.")

    non_role = await call(context, op="reset", name="reviewer")

    assert "local-operator agents delete --name reviewer" in non_role
    assert "conversation history" in non_role, "say what the escape costs"
    assert "Rename that agent" not in non_role


@pytest.mark.asyncio
async def test_an_absent_packaged_effort_does_not_render_as_a_tier(context) -> None:
    """U11: `packaged effort: (unset)` invites the reader to parse `(unset)` as
    a literal tier."""
    _legacy_role_row(context.agent_registry, "manager", instructions="MY OWN REWRITE")
    await call(context, op="update", name="manager", effort="lo")

    body = await call(context, op="show", name="manager")

    assert "packaged effort: not set by the starter" in body
    assert "packaged effort: (unset)" not in body


@pytest.mark.asyncio
async def test_a_description_only_edit_is_visible_and_resettable(context, registry) -> None:
    """F6: `install_seed` writes `description`, but `seed_divergence` excluded
    it, so a description-only edit was invisible and unresettable — F1's bug one
    field over. The exclusion was justified by a claim that comparing it would
    flag every installed role; that is true only against `seed.description`,
    which is NOT the value install writes."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", description="does something else entirely")

    shown = await call(context, op="show", name="reviewer")
    body = await call(context, op="reset", name="reviewer")

    assert "description differs" in shown
    assert "description replaced" in body
    assert "does something else entirely" in body, "the routing text must be echoed, not dropped"
    packaged = load_seed("reviewer")
    profile = resolve_profile("reviewer", registry=registry)
    assert packaged is not None and profile is not None
    assert profile.description.strip() == (packaged.when_to_use or packaged.description).strip()


@pytest.mark.parametrize("name", ["architect", "coder", "designer", "manager", "reviewer", "scout"])
def test_a_freshly_installed_starter_reports_no_divergence(tmp_path, name: str) -> None:
    """F6, run rather than reasoned: the guard against the exclusion's stated
    justification. If comparing `description` really did flag every installed
    role, this fails for all six — it flags none."""
    from local_operator.agent_profiles import (
        install_seed,
        profile_from_agent,
        seed_divergence,
    )

    registry = AgentRegistry(tmp_path)
    install_seed(name, registry=registry)
    seed = load_seed(name)
    assert seed is not None
    row = registry.get_agent_by_name(name)
    assert row is not None
    profile = profile_from_agent(registry, row)

    assert seed_divergence(profile, seed) == ()


@pytest.mark.asyncio
async def test_search_still_finds_a_role_after_it_is_reset(context) -> None:
    """F6's observable effect: `description` is the text `search` embeds, so a
    reset that rewrote it without comparing it broke the user's routing."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="broken")
    await call(context, op="reset", name="reviewer")

    body = await call(context, op="search", query="check this pull request diff for bugs")

    assert "reviewer" in body


def test_a_provenance_marker_naming_another_role_is_ignored(tmp_path) -> None:
    """F8: tags are writable by the server routes, the desktop UI and agent
    import, none of which know what this marker means. A destructive verb keying
    on it must not trust a `seed:reviewer` tag found on some other agent."""
    from local_operator.agent_profiles import install_seed, seed_origin

    registry = AgentRegistry(tmp_path)
    install_seed("reviewer", registry=registry)
    agent = registry.get_agent_by_name("reviewer")
    assert agent is not None
    assert seed_origin(agent) == "reviewer"

    def origin_now() -> str | None:
        row = registry.get_agent_by_name("reviewer")
        assert row is not None
        return seed_origin(row)

    registry.update_agent(agent.id, _edit_fields(tags=["role", "seed:coder"]))
    assert origin_now() is None

    registry.update_agent(agent.id, _edit_fields(tags=["role", "seed:not-a-starter"]))
    assert origin_now() is None


@pytest.mark.asyncio
async def test_a_forged_marker_cannot_make_reset_overwrite_someone_elses_work(
    context, registry
) -> None:
    """F8, the consequence rather than the predicate: a cross-name marker must
    leave a self-authored role protected."""
    await call(context, op="create", name="scout", description="mine", instructions="MY OWN WORK")
    agent = registry.get_agent_by_name("scout")
    assert agent is not None
    registry.update_agent(agent.id, _edit_fields(tags=["role", "seed:reviewer"]))

    body = await call(context, op="reset", name="scout")

    assert "no record of being installed" in body
    profile = resolve_profile("scout", registry=registry)
    assert profile is not None and profile.instructions == "MY OWN WORK"


@pytest.mark.asyncio
async def test_reinstall_offers_reset_only_where_reset_would_work(context) -> None:
    """F10: `show` was gated on provenance and this message was not, so a legacy
    edited row was still sent to a verb that declines — the #141 dead end
    surviving in a second message."""
    _legacy_role_row(context.agent_registry, "reviewer", instructions="MY OWN REWRITE")

    body = await call(context, op="install", name="reviewer")

    assert "left as-is" in body
    assert "op='reset'" not in body, "reset would refuse this row"
    assert "op='update' name='reviewer'" in body


@pytest.mark.asyncio
async def test_reinstall_still_offers_reset_on_a_provenanced_row(context) -> None:
    """The F10 gate must not remove the offer where it does work."""
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions="edited")

    body = await call(context, op="install", name="reviewer")

    assert "op='reset' name='reviewer'" in body


@pytest.mark.asyncio
async def test_the_non_role_refusal_names_an_escape_that_runs(context, registry) -> None:
    """U12: it said "install the starter under a different name", which cannot
    work — `install` takes only packaged names and nothing renames — leaving a
    history-destroying shell delete as the only named route on the one path that
    has a safe in-tool answer."""
    agent = registry.create_agent(_edit_fields(name="reviewer", description="my chat agent"))
    registry.set_agent_system_prompt(agent.id, "Be agreeable.")

    body = await call(context, op="reset", name="reviewer")

    assert "Install the starter under a different name" not in body
    assert "op='show' name='reviewer'" in body and "op='create' name='my-reviewer'" in body
    # The escape it names must actually run for this reader.
    assert "packaged starter" in await call(context, op="show", name="reviewer")
    created = await call(
        context, op="create", name="my-reviewer", description="d", instructions="pasted"
    )
    assert "created role 'my-reviewer'" in created
    assert registry.get_agent_system_prompt(agent.id) == "Be agreeable.", "must not overwrite"


@pytest.mark.asyncio
async def test_show_prints_what_it_tells_an_unrecorded_reader_to_apply(context) -> None:
    """U13: the closing line said "apply the packaged values yourself" while the
    body rendered a DIFF, which carries only the changed lines and cannot be
    applied. What the reader is told to do decides what they are shown.

    The fixture must be a SMALL edit to the packaged body, not a short unrelated
    string. `_instruction_diff` falls back to the full body whenever the diff
    would not be shorter, so a one-line instruction reaches that fallback on its
    own and passes this test no matter what the provenance branch does —
    pinning nothing. This seeds the packaged text with one line changed, which
    is exactly the input that makes a diff the *cheaper* rendering and so forces
    the branch under test to be the thing that decides.
    """
    packaged = load_seed("manager")
    assert packaged is not None
    lines = packaged.instructions.strip().splitlines()
    edited = "\n".join([lines[0], "I ALSO IMPLEMENT WHEN I FEEL LIKE IT.", *lines[1:]])
    _legacy_role_row(context.agent_registry, "manager", instructions=edited)

    body = await call(context, op="show", name="manager")

    assert "packaged instructions in full" in body
    assert "as a diff against yours" not in body
    missing = [
        line
        for line in packaged.instructions.strip().splitlines()
        if line.strip() and line not in body
    ]
    assert not missing, f"the reader cannot apply what is not printed: {missing[:2]}"


@pytest.mark.asyncio
async def test_show_still_diffs_a_row_that_reset_can_restore(context) -> None:
    """The U13 change is scoped to the hand-apply path: where `reset` does the
    applying, the diff is still the useful rendering."""
    packaged = load_seed("reviewer")
    assert packaged is not None
    lines = packaged.instructions.splitlines()
    tweaked = "\n".join(lines[:5] + ["You may skip the tests if you are in a hurry."] + lines[5:])
    await call(context, op="install", name="reviewer")
    await call(context, op="update", name="reviewer", instructions=tweaked)

    body = await call(context, op="show", name="reviewer")

    assert "as a diff against yours" in body
