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
    assert "no registered role" in await call(context, op="update", name="ghost", instructions="x")


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
    """C1, the same bug on the other two role fields."""
    await call(context, op="install", name="manager")
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
