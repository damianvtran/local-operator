"""The ``team`` tool: discover, create, and refine reusable teams.

A team is a named roster (manager + members) plus two instruction layers
that do not belong on any one agent: how the group collaborates, and which
product or domain this instance of the group is responsible for. Agents stay
reusable; the team is what specialises them for a grouping.

See :mod:`local_operator.teams` for storage and the layering contract, and
``guide://teams`` for how an agent should work with a user to author one.

CONTEXT DISCIPLINE
==================

The registry is NEVER enumerated into the prompt. ``list`` returns one line
per team and only when called; ``show`` is the only op that returns the
full briefs. Mirrors how roles and skills are handled.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.teams import (
    MAX_TEAM_INSTRUCTIONS_CHARS,
    TeamEditFields,
    TeamRegistry,
    TeamRegistryLockTimeout,
    parse_members,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    spill_truncate,
)

logger = logging.getLogger(__name__)

_ROW_CAP = 160


class TeamParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "create", "update"] = Field(
        description=(
            "list/show: what teams exist and what they say; create/update: "
            "author or fix a team. Use team_delete for irreversible removal."
        )
    )
    name: str | None = Field(default=None, description="Team name (all ops but list).")
    description: str | None = Field(
        default=None,
        description="create/update: one line on what this team is for.",
    )
    manager: str | None = Field(
        default=None,
        description=(
            "create/update: role or specialist who orchestrates. Defaults to "
            "'manager'. Install that starter if the user has not authored one."
        ),
    )
    members: list[str] | None = Field(
        default=None,
        description=(
            "create/update: roster slots as 'role' or 'role:count'. "
            "The same role can sit on many teams; counts spawn copies. "
            "Prefix a slot with 'team:' to nest ANOTHER team as a sub-org — "
            "'team:pod' or 'team:pod:2' (name 'pod', two copies). A nested "
            "team carries its own briefs; the slot only points at it by name."
        ),
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "create/update: how this team collaborates. Layered ON TOP of "
            "each member's own instructions, never a replacement."
        ),
    )
    project: str | None = Field(
        default=None,
        description=(
            "create/update: the product or domain this instance of the team "
            "is responsible for. Swap this to reuse the same roster on "
            "another product."
        ),
    )


def _registry(context: ToolContext | None) -> TeamRegistry | None:
    raw = getattr(context, "team_registry", None) if context else None
    return raw if isinstance(raw, TeamRegistry) else None


def _row(team: Any) -> str:
    slots = len(team.members) + 1
    summary = (team.description or "").strip() or "(no description)"
    role_word = "role" if slots == 1 else "roles"
    row = f"- {team.name} [{slots} {role_word}, led by {team.manager}]: {summary}"
    return row if len(row) <= _ROW_CAP else row[: _ROW_CAP - 1].rstrip() + "…"


async def _op_list(context: ToolContext | None, tool_call_id: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "team", "no team registry attached to this session.")
    teams = registry.list_teams()
    if not teams:
        body = (
            "no teams registered. Create one with op='create' after agreeing "
            "the manager, the roster, how they collaborate, and (if this "
            "instance owns a product) the project brief. Read guide://teams first."
        )
        return _text(tool_call_id, "team", body)
    body = "registered teams:\n" + "\n".join(_row(team) for team in teams)
    body += "\n\nlaunch with /team <name> <request>, or op='show' name='<name>'."
    text, spill = spill_truncate(body, "team", context)
    return _text(tool_call_id, "team", text, details=spill or None)


async def _op_show(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "team", "no team registry attached to this session.")
    team = registry.get_team_by_name(name)
    if team is None:
        return _error(tool_call_id, "team", f"no team named {name!r} (try op='list')")
    header = [
        f"{team.name}. Led by {team.manager}",
        f"description: {team.description or '(unstated)'}",
        "roster:",
        *team.roster_lines(),
    ]
    body = "\n".join(header)
    if team.instructions.strip():
        body += "\n\ncollaboration:\n" + team.instructions.strip()
    else:
        body += "\n\ncollaboration: (none)"
    if team.project.strip():
        body += "\n\nproject:\n" + team.project.strip()
    else:
        body += "\n\nproject: (none)"
    body += f"\n\nlaunch with /team {team.name} <request>."
    text, spill = spill_truncate(body, "team", context)
    return _text(tool_call_id, "team", text, details=spill or None)


def _bounded_field(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if len(value) > MAX_TEAM_INSTRUCTIONS_CHARS:
        raise ValueError(
            f"{label} exceed {MAX_TEAM_INSTRUCTIONS_CHARS} chars; they ride in "
            "front of every run of this team, so they must stay short."
        )
    return value


async def _op_write(
    context: ToolContext | None,
    tool_call_id: str,
    params: TeamParams,
    *,
    creating: bool,
) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(
            tool_call_id, "team", "no team registry attached to this session; cannot save teams."
        )
    name = (params.name or "").strip()
    existing = registry.get_team_by_name(name)
    if creating and existing is not None:
        return _error(
            tool_call_id,
            "team",
            f"team {name!r} already exists; use op='update' to change it.",
        )
    if not creating and existing is None:
        return _error(tool_call_id, "team", f"no team named {name!r} to update.")
    try:
        members = parse_members(params.members) if params.members is not None else None
        instructions = _bounded_field(params.instructions, "collaboration instructions")
        project = _bounded_field(params.project, "project instructions")
    except ValueError as exc:
        return _error(tool_call_id, "team", str(exc))

    fields = TeamEditFields(
        name=name if creating else None,
        description=params.description,
        manager=params.manager,
        members=members,
        instructions=instructions,
        project=project,
    )
    try:
        if creating:
            if not (params.manager or "").strip():
                fields.manager = "manager"
            team = registry.create_team(fields)
            verb = "created"
        else:
            assert existing is not None  # guarded above
            team = registry.update_team(existing.id, fields)
            verb = "updated"
    except TeamRegistryLockTimeout as exc:
        # U5-1: contention is recoverable, so the model gets the same concise
        # guidance the CLI prints rather than a generic "could not save".
        return _error(tool_call_id, "team", str(exc))
    except ValueError as exc:
        return _error(tool_call_id, "team", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "team", f"could not save team: {exc}")
    return _text(
        tool_call_id,
        "team",
        f"{verb} team {team.name!r}. {team.manager} leads it; "
        f"{len(team.members)} member slot(s). Launch with /team {team.name} <request>.",
    )


async def _op_delete(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "team_delete", "no team registry attached to this session.")
    team = registry.get_team_by_name(name)
    if team is None:
        return _error(tool_call_id, "team_delete", f"no team named {name!r} to delete.")
    try:
        registry.delete_team(team.id)
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "team_delete", f"could not delete team: {exc}")
    return _text(tool_call_id, "team_delete", f"deleted team {name!r}.")


@_guard("team")
async def execute_team(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Discover and author reusable teams."""
    try:
        params = TeamParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "team", exc)

    if params.op in {"show", "create", "update"} and not (params.name or "").strip():
        return _error(tool_call_id, "team", f"op={params.op!r} needs 'name'.")
    if params.op == "list":
        return await _op_list(context, tool_call_id)
    if params.op == "show":
        return await _op_show(context, tool_call_id, str(params.name))
    return await _op_write(context, tool_call_id, params, creating=params.op == "create")


def build_team_tool(context: ToolContext) -> AgentTool | None:
    """createIf: the tool exists only where a registry can back it."""
    if getattr(context, "team_registry", None) is None:
        return None
    return AgentTool(
        name="team",
        label="Teams",
        description=(
            "Reusable teams: a manager plus members (roles or specialists, "
            "with counts), plus collaboration and project briefs layered on "
            "top of each member's own instructions. Find or author one; use "
            "team_delete for irreversible removal. The user launches it with "
            "/team <name> <request>."
        ),
        parameters=TeamParams.model_json_schema(),
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_team,
    )


class TeamDeleteParams(BaseModel):
    """Separate schema so irreversible removal can carry a write-tier gate."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Existing team name to delete permanently.")


@_guard("team_delete")
async def execute_team_delete(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Delete one team after the loop has passed the write approval gate."""
    try:
        params = TeamDeleteParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "team_delete", exc)
    name = params.name.strip()
    if not name:
        return _error(tool_call_id, "team_delete", "name must not be empty.")
    return await _op_delete(context, tool_call_id, name)


def build_team_delete_tool(context: ToolContext) -> AgentTool | None:
    """createIf: destructive deletion exists only beside a real registry."""
    if getattr(context, "team_registry", None) is None:
        return None
    return AgentTool(
        name="team_delete",
        label="Delete team",
        description=(
            "Permanently remove a saved team. This is separate from the team "
            "authoring tool so deletion always asks for write approval."
        ),
        parameters=TeamDeleteParams.model_json_schema(),
        approval_tier="write",
        concurrency="exclusive",
        interruptible=False,
        describe_approval=lambda args, _cwd: (
            f"Delete team {str(args.get('name') or '').strip()!r} permanently"
        ),
        execute=execute_team_delete,
    )
