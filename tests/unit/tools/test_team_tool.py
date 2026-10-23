"""The ``team`` tool: create, show, update, delete."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.teams import TeamEditFields, TeamRegistry
from local_operator.tools.team_tool import (
    build_team_delete_tool,
    build_team_tool,
    execute_team,
    execute_team_delete,
)


@pytest.fixture()
def registry(tmp_path) -> Any:
    """The SESSION's composite (see ``test_agent_tool.py::registry``)."""
    from local_operator.profile_scope import scope_team_registry

    return scope_team_registry(TeamRegistry(tmp_path), tmp_path / "session")


@pytest.fixture()
def context(registry: TeamRegistry) -> ToolContext:
    return ToolContext(cwd=".", team_registry=registry)


async def call(context: ToolContext, **args: Any) -> str:
    result = await execute_team("tc", args, None, None, context)
    return result.text


async def delete(context: ToolContext, name: str) -> str:
    result = await execute_team_delete("tc", {"name": name}, None, None, context)
    return result.text


def test_the_tool_is_not_advertised_without_a_registry() -> None:
    assert build_team_tool(ToolContext(cwd=".")) is None
    assert build_team_delete_tool(ToolContext(cwd=".")) is None
    assert build_team_tool(ToolContext(cwd=".", team_registry=object())) is not None
    assert build_team_delete_tool(ToolContext(cwd=".", team_registry=object())) is not None


def test_only_destructive_team_removal_requires_approval(context) -> None:
    authoring = build_team_tool(context)
    deleting = build_team_delete_tool(context)
    assert authoring is not None and authoring.approval_tier == "read"
    assert deleting is not None and deleting.approval_tier == "write"
    assert deleting.describe_approval is not None
    assert "Delete team 'alpha' permanently" == deleting.describe_approval({"name": "alpha"}, ".")


@pytest.mark.asyncio
async def test_list_empty_points_at_create(context) -> None:
    body = await call(context, op="list")
    assert "no teams" in body
    assert "guide://teams" in body


@pytest.mark.asyncio
async def test_create_show_update_delete(context) -> None:
    created = await call(
        context,
        op="create",
        name="feature-release",
        description="Ship a change",
        manager="manager",
        members=["coder", "reviewer:2"],
        instructions="Review before merge.",
        project="user-dashboard",
    )
    assert "created team 'feature-release'" in created
    shown = await call(context, op="show", name="feature-release")
    assert "Led by manager" in shown
    assert "coder" in shown
    assert "Review before merge." in shown
    assert "user-dashboard" in shown
    updated = await call(
        context,
        op="update",
        name="feature-release",
        project="admin-api",
        members=["designer", "reviewer:2"],
    )
    assert "updated team" in updated
    shown_again = await call(context, op="show", name="feature-release")
    assert "admin-api" in shown_again
    assert "Review before merge." in shown_again
    assert "designer" in shown_again
    assert "reviewer x2" in shown_again
    deleted = await delete(context, "feature-release")
    assert "deleted team" in deleted
    assert "no team named" in await call(context, op="show", name="feature-release")


@pytest.mark.asyncio
async def test_create_refuses_a_duplicate(context) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="create", name="alpha")
    assert "already exists" in body


@pytest.mark.asyncio
async def test_create_needs_a_name(context) -> None:
    assert "needs 'name'" in await call(context, op="create")


@pytest.mark.asyncio
async def test_lock_timeout_returns_structured_error_without_traceback(
    tmp_path, monkeypatch
) -> None:
    """U5-1: contention through the tool path is a recoverable error result.

    A peer holds the registry lock past the bounded wait; the model must get
    the retry guidance as a structured error, not a raised exception (the
    loop never raises into the model) and not a generic could-not-save.
    """
    import fcntl
    import os

    import local_operator.teams as teams_module
    from local_operator.teams import TeamEditFields, TeamRegistry

    cfg = tmp_path
    holder = TeamRegistry(cfg)
    holder.create_team(TeamEditFields(name="seed"))
    monkeypatch.setattr(teams_module, "_TEAM_LOCK_TIMEOUT_S", 0.2)

    # The tool's default is a CONVERSATION-scoped team, so the contended store is
    # the session one (``<session>/roles``); the global lock is not touched.
    from local_operator.profile_scope import scope_team_registry

    registry = scope_team_registry(TeamRegistry(cfg), cfg / "session")
    context = ToolContext(cwd=".", team_registry=registry)
    (cfg / "session" / "roles").mkdir(parents=True, exist_ok=True)
    fd = os.open(cfg / "session" / "roles" / ".teams.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        result = await execute_team("tc", {"op": "create", "name": "blocked"}, None, None, context)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert result.is_error
    text = result.text
    assert "Timed out waiting for the teams registry lock" in text
    assert "retry after the other lop process finishes" in text
    assert "could not save team" not in text


def test_team_params_carry_label_and_aliases_and_refuse_unknown_fields() -> None:
    from pydantic import ValidationError

    from local_operator.tools.team_tool import TeamParams

    params = TeamParams(op="create", name="x", label="Display", aliases=["dx"])
    assert params.label == "Display" and params.aliases == ["dx"]
    # Unknown fields are refused (constructed from a dict: the point is the
    # MODEL's extra="forbid", so the typo must not fail a type check first).
    with pytest.raises(ValidationError):
        TeamParams.model_validate({"op": "create", "name": "x", "labell": "typo"})


@pytest.mark.asyncio
async def test_create_renders_label_first_and_aliases_resolve(context) -> None:
    created = await call(
        context,
        op="create",
        name="data-quality",
        label="Data Quality",
        aliases=["dq"],
        description="Finds issues",
    )
    assert "created team 'data-quality'" in created  # the receipt stays key-based

    listed = await call(context, op="list")
    # The shared display rule: the derived default paints alone.
    assert "Data Quality" in listed
    assert "(data-quality)" not in listed

    # The alias resolves; the header paints the shared form and the aliases
    # line names the keys the model can address (D5).
    shown = await call(context, op="show", name="dq")
    assert "Data Quality. Led by manager" in shown
    assert "aliases: dq" in shown
    assert "launch with /team data-quality <request>" in shown


@pytest.mark.asyncio
async def test_create_whose_name_is_another_team_s_alias_says_so(context) -> None:
    """Q1: the collision's own wording, not the generic pre-check sentence."""
    await call(context, op="create", name="alpha", aliases=["a1"])
    body = await call(context, op="create", name="a1")
    assert "Team name 'a1' is already an alias of team 'alpha'" in body
    # A plainly taken NAME keeps the tool's actionable message.
    taken = await call(context, op="create", name="alpha")
    assert "already exists; use op='update'" in taken


@pytest.mark.asyncio
async def test_update_resets_a_label_and_replaces_aliases(context, registry) -> None:
    await call(context, op="create", name="ops", label="Ops Display", aliases=["o1"])
    updated = await call(context, op="update", name="ops", label="", aliases=["o2"])
    assert "updated team" in updated
    team = registry.get_team_by_name("ops")
    assert team is not None
    assert team.label == "Ops"  # reset to the derived default
    assert team.aliases == ["o2"]  # replaced wholesale, not merged


# ---------------------------------------------------------------------------
# LOH-9 S2 — scope on writes, consent for the account store, session deletes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tool_create_defaults_to_this_conversation(context, registry) -> None:
    body = await call(context, op="create", name="pod", manager="manager", members=["scout"])

    assert "in this conversation" in body
    assert registry.global_registry.get_team_by_name("pod") is None
    assert registry.store_for("session").get_team_by_name("pod") is not None

    result = await execute_team(
        "tc", {"op": "create", "name": "pod-2", "manager": "manager"}, None, None, context
    )
    assert result.details is not None
    assert result.details["scope"] == "session"
    assert result.details["authorized_by"] is None


@pytest.mark.asyncio
async def test_a_tool_cannot_create_in_the_account_without_the_host(context, registry) -> None:
    body = await call(context, op="create", name="durable", scope="global", manager="manager")

    assert "needs the user's confirmation" in body
    assert "TUI `/team` form" in body
    assert registry.global_registry.get_team_by_name("durable") is None


@pytest.mark.asyncio
async def test_a_host_approved_account_team_records_its_origin(registry) -> None:
    seen: list[Any] = []

    async def approve(intent):
        seen.append(intent)
        return "ask:7"

    context = ToolContext(cwd=".", team_registry=registry, authorize_durable_write=approve)
    result = await execute_team(
        "tc",
        {"op": "create", "name": "durable", "scope": "global", "manager": "manager"},
        None,
        None,
        context,
    )

    assert not result.is_error
    assert "in your account" in result.text
    assert result.details is not None
    assert result.details["authorized_by"] == "ask:7"
    assert seen and seen[0].kind == "team"
    assert registry.global_registry.get_team_by_name("durable") is not None


@pytest.mark.asyncio
async def test_delete_removes_a_conversation_team_only(context, registry) -> None:
    await call(context, op="create", name="pod", manager="manager", members=["scout"])

    body = await delete(context, "pod")

    assert "from this conversation" in body
    assert "account is unchanged" in body
    assert registry.get_team_by_name("pod") is None


@pytest.mark.asyncio
async def test_delete_refuses_an_account_team(context, registry) -> None:
    registry.global_registry.create_team(
        TeamEditFields(name="durable", manager="manager", members=[])
    )

    body = await delete(context, "durable")

    assert "saved in the user's account" in body
    assert registry.global_registry.get_team_by_name("durable") is not None


# ---------------------------------------------------------------------------
# Round 1 remediation — alias collisions, grouping and scope in listings
# ---------------------------------------------------------------------------


def _consenting_team(registry, origin: str = "test-consent"):
    seen: list[Any] = []

    async def approve(intent):
        seen.append(intent)
        return origin

    return (
        ToolContext(cwd=".", team_registry=registry, authorize_durable_write=approve),
        seen,
    )


@pytest.mark.asyncio
async def test_a_session_create_over_an_account_alias_refuses_before_consent(registry) -> None:
    """M1/R-m2: an alias resolves like a name, so it shadows like one."""
    registry.global_registry.create_team(
        TeamEditFields(name="release-train", manager="manager", members=[], aliases=["release"])
    )
    context, seen = _consenting_team(registry)

    body = await call(context, op="create", name="release", manager="manager", members=[])

    assert "already exists in your account" in body
    assert seen == [], "the alias collision must not reach the consent question"


@pytest.mark.asyncio
async def test_list_groups_by_scope_and_show_names_it(context, registry) -> None:
    """U1/AC13: a roster choice must say which store a team lives in."""
    await call(context, op="create", name="here-team", manager="manager", members=[])
    registry.global_registry.create_team(
        TeamEditFields(name="everywhere-team", manager="manager", members=[])
    )

    listed = await call(context, op="list")

    assert "teams in this conversation:" in listed
    assert "teams saved everywhere (your account):" in listed
    shown = await call(context, op="show", name="here-team")
    assert "scope: this conversation" in shown


# ---------------------------------------------------------------------------
# LOH-9 S4 — promotion: the refusal, the grant, and the member receipt
# ---------------------------------------------------------------------------


def _promote_context(registry, **kwargs):
    """A context carrying BOTH registries, as a real session's tool context does."""

    from local_operator.agents import AgentRegistry
    from local_operator.profile_scope import scope_agent_registry

    agents = scope_agent_registry(AgentRegistry(registry.config_dir), registry.scope_dir.parent)
    return ToolContext(cwd=".", team_registry=registry, agent_registry=agents, **kwargs)


@pytest.mark.asyncio
async def test_a_tool_team_promote_is_refused_without_a_host_grant(registry) -> None:
    context = _promote_context(registry)
    await call(context, op="create", name="pod", manager="manager", members=[])

    result = await execute_team("tc", {"op": "promote", "name": "pod"}, None, None, context)

    assert "this run has no way to ask" in result.text
    assert "/team promote pod" in result.text
    assert registry.global_registry.get_team_by_name("pod") is None


@pytest.mark.asyncio
async def test_a_granted_team_promote_lists_the_members_it_copied(registry) -> None:
    seen: list[Any] = []

    async def approve(intent):
        seen.append(intent)
        return "test-consent"

    from local_operator.agents import AgentEditFields

    context = _promote_context(registry, authorize_durable_write=approve)
    # A conversation role the roster needs, plus the team that names it.
    role = context.agent_registry.create_agent(
        # Every field spelled out: ``AgentEditFields`` is validated in strict mode,
        # and a partially-spelled construction is a pyright error even though the
        # rest default to ``None``.
        AgentEditFields(
            name="helper",
            label=None,
            description=None,
            tags=["role"],
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
        ),
        scope="session",
    )
    context.agent_registry.set_agent_system_prompt(role.id, "Help.\n")
    await call(context, op="create", name="pod", manager="helper", members=["helper"])

    result = await execute_team("tc", {"op": "promote", "name": "pod"}, None, None, context)

    assert "copied team 'pod' to your account" in result.text
    assert "It also copied what the roster needs: helper." in result.text
    assert result.details is not None
    assert result.details["expanded"] == ["helper"]
    assert result.details["authorized_by"] == "test-consent"
    assert [(intent.kind, intent.operation) for intent in seen] == [("team", "promote")]
    assert registry.global_registry.get_team_by_name("pod") is not None
    assert context.agent_registry.global_registry.get_agent_by_name("helper") is not None


@pytest.mark.asyncio
async def test_a_tool_team_promote_needs_the_agent_registry(registry) -> None:
    """Without the agent composite the roster cannot be resolved, so it refuses."""

    context = ToolContext(cwd=".", team_registry=registry)
    await call(context, op="create", name="pod", manager="manager", members=[])

    result = await execute_team("tc", {"op": "promote", "name": "pod"}, None, None, context)

    assert "agent registry" in result.text
    assert registry.global_registry.get_team_by_name("pod") is None
