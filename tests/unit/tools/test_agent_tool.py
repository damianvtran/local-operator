"""The ``agent`` tool: role discovery, installation, and authoring."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.agent_profiles import resolve_profile
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
    assert "5 tools" in rows["scout"], rows["scout"]


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
