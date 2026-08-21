"""The ``team`` tool: create, show, update, delete."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.teams import TeamRegistry
from local_operator.tools.team_tool import (
    build_team_delete_tool,
    build_team_tool,
    execute_team,
    execute_team_delete,
)


@pytest.fixture()
def registry(tmp_path) -> TeamRegistry:
    return TeamRegistry(tmp_path)


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
