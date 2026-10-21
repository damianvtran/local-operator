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
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    DurableWriteIntent,
    ToolContext,
    ToolResult,
)
from local_operator.projects import truncate_row
from local_operator.teams import (
    MAX_TEAM_INSTRUCTIONS_CHARS,
    TeamEditFields,
    TeamMember,
    TeamRegistryLockTimeout,
    TeamRegistryRecoveryError,
    display_form,
    find_member,
    parse_members,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    identifier_text,
    spill_truncate,
)

logger = logging.getLogger(__name__)

_ROW_CAP = 160


class TeamOverride(BaseModel):
    """One ``overrides`` entry: a roster slot, and the fields to set or clear."""

    model_config = ConfigDict(extra="forbid")

    member: str = Field(
        description=(
            "The roster slot to change, spelled exactly as in 'members': a role "
            "name, or 'team:<name>' for a nested-team slot."
        )
    )
    tier: str | None = Field(
        default=None,
        description=(
            "Which model tier this slot runs on: 'lo', 'med' or 'hi' by default "
            "(a key of values.subagents.models). 'inherit' clears it, so the slot "
            "falls back to the role's own effort tag. Omit to leave it alone."
        ),
    )
    effort: str | None = Field(
        default=None,
        description=(
            "How much reasoning this slot asks for: 'auto' (the model's own "
            "default) or a level from 'none' to 'max', clamped to the model's "
            "ladder. 'inherit' clears it. Omit to leave it alone."
        ),
    )


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
    label: str | None = Field(
        default=None,
        description=(
            "create/update: the display name for listings, the chart, the "
            "picker and the status band (free text, spaces allowed). The team "
            "is still ADDRESSED by 'name'. On update, an empty string resets "
            "the label to its derived default."
        ),
    )
    aliases: list[str] | None = Field(
        default=None,
        description=(
            "create/update: EXTRA addressing keys this team also answers to "
            "(letters, digits, dot, underscore or hyphen; up to 8). The whole "
            "list replaces any stored one; each key must not collide with a "
            "team name or another team's alias."
        ),
    )
    scope: Literal["global", "session"] | None = Field(
        default=None,
        description=(
            "'session' (default) is this conversation only; 'global' is the "
            "user's account and needs their confirmation."
        ),
    )
    overrides: list[TeamOverride] | None = Field(
        default=None,
        description=(
            "create/update: pin the MODEL or the REASONING EFFORT of individual "
            "roster slots, so one member of a team can run on a stronger model "
            "than the rest. Each entry names a slot from 'members' and sets or "
            "clears 'tier'/'effort'; a slot this list does not name keeps "
            "whatever it already had, including across an update that replaces "
            "'members'. On a nested-team slot ('team:pod') the pin belongs to "
            "that team's MANAGER when this roster launches it. Only available "
            "when subagents.model_choice is 'model'; otherwise the operator "
            "sets these in the desktop team editor."
        ),
    )


def _slot_token(member: TeamMember) -> str:
    """How ``members`` spells this slot, for a message a caller can act on."""

    return f"team:{member.role}" if member.kind == "team" else str(member.role)


def _override_lines(team: Any, mark: Callable[[str], str] | None = None) -> list[str]:
    """The lines ``show`` adds for a roster carrying member overrides.

    Deliberately NOT in ``roster_lines()``: that feeds the manager's and every
    member's prompt, and a prompt change is a prompt-cache cost for no behaviour
    gain. ``show`` is the surface a model reads when it needs to know, and the
    desktop editor is the surface that writes them.

    ``mark`` wraps the slot name for the redaction pass, like
    ``Team.roster_lines``; a slot name is a rendered identifier.
    """

    label = mark or (lambda value: value)
    rows: list[str] = []
    for member in team.members:
        if member.is_default():
            continue
        parts = []
        if member.tier:
            parts.append(f"tier {member.tier}")  # i18n: ignore model-facing tool output
        if member.effort:
            parts.append(f"effort {member.effort}")  # i18n: ignore model-facing tool output
        rows.append(f"- {label(_slot_token(member))}: {', '.join(parts)}")
    if not rows:
        return []
    return ["member overrides (model tier / reasoning effort):", *rows]


def _carry_overrides(existing: Any, members: list[TeamMember]) -> list[TeamMember]:
    """Keep the stored overrides on every slot ``members`` still names.

    An ``update`` with ``members=`` replaces the roster WHOLESALE, so a roster
    edit -- adding one role, reordering, or an editor echoing back what it read --
    would otherwise drop every override the operator had set. They are PINS: a
    dropped one puts that child back on the model the operator moved it off, and
    nothing in the receipt would say so.
    """

    if existing is None:
        return members
    carried: list[TeamMember] = []
    for member in members:
        stored = find_member(existing.members, member.role, kind=member.kind)
        if stored is not None and not stored.is_default() and member.is_default():
            member = member.model_copy(update={"tier": stored.tier, "effort": stored.effort})
        carried.append(member)
    return carried


def _apply_overrides(members: list[TeamMember], overrides: list[TeamOverride]) -> list[TeamMember]:
    """Set or clear the named slots' overrides, in place, on a fresh roster.

    ``inherit`` CLEARS a field, the same word the ``agent`` tool uses to clear a
    pin, so a caller can take an override off without re-sending the roster.
    A slot the roster does not name is refused by name rather than ignored: the
    caller asked for something that did not happen, and silence about that is
    how an operator ends up believing a pin exists.
    """

    updated = list(members)
    for override in overrides:
        token = (override.member or "").strip()
        kind = "team" if token.lower().startswith("team:") else "agent"
        name = token[len("team:") :].strip() if kind == "team" else token
        index = next(
            (
                i
                for i, member in enumerate(updated)
                if member.kind == kind and str(member.role).strip().casefold() == name.casefold()
            ),
            None,
        )
        if index is None:
            roster = ", ".join(_slot_token(member) for member in updated) or "(empty)"
            raise ValueError(
                f"no roster slot named {token!r} to put overrides on; this roster is: {roster}"
            )
        fields: dict[str, Any] = {}
        for field in ("tier", "effort"):
            value = getattr(override, field)
            if value is None:
                continue
            fields[field] = None if value.strip().lower() == "inherit" else value.strip().lower()
        if fields:
            # Validate through the MODEL, not ``model_copy(update=...)``: the
            # copy skips the field validators, so a bad word used to reach
            # ``team.yml`` and the next load dropped it -- the receipt claimed a
            # pin that did not exist, and the child then ran on a model the
            # operator was not told about. The registry re-validates at save for
            # the transports that never come through here; this arm is for the
            # caller that can still correct its call.
            current = updated[index]
            updated[index] = TeamMember.model_validate({**current.model_dump(), **fields})
    return updated


def _overrides_refusal() -> str:
    """Why a model may not set member overrides, and who can.

    Ordered the way :func:`local_operator.tools.builtin._operator_choice_pin_rejection`
    is: the OWNERSHIP first, then the operator's route, then the model's own
    next step — the truncated card shows that lead and nothing else.
    """

    return (
        "Member overrides are the operator's: they pin a slot's model or its "
        "reasoning effort. Raising subagents.model_choice to 'model' delegates "
        "that choice; until then the operator sets them in the desktop team "
        "editor. Author the roster without 'overrides'."
    )


def _registry(context: ToolContext | None) -> Any | None:
    """The session's team authority, duck-typed.

    A session holds a COMPOSITE over its scoped store and the global one
    (LOH-9), so an ``isinstance(TeamRegistry)`` gate would answer "no registry"
    in every session while the tool was still advertised. The two methods this
    module calls on it are the contract; anything answering them is a registry.
    """
    raw = getattr(context, "team_registry", None) if context else None
    if raw is None or not hasattr(raw, "list_teams") or not hasattr(raw, "get_team_by_name"):
        return None
    return raw


def _row(team: Any, *, context: ToolContext | None = None) -> str:
    # ``member_count()`` for the same reason every other listing uses it (D2):
    # `len(members) + 1` assumes the manager is not on the roster, which is
    # false for real teams, and collapses a multi-count slot to one. This row
    # is what the MODEL reads when it lists teams, so a wrong count here is a
    # wrong count in the agent's own reasoning about a roster it may staff.

    # Identifier fields: the display form and the manager are marked spans; a
    # team may be named or led by a word that looks credential-shaped.
    slots = team.member_count()
    summary = (team.description or "").strip() or "(no description)"
    # "members", matching the number (R6): `member_count()` excludes the
    # manager, whom this row names separately, so "roles" implied he was
    # counted and put the model one ahead of the roster it can actually staff.
    member_word = "member" if slots == 1 else "members"
    # The shared display rule (D2): this row is what the MODEL reads when it
    # lists teams, and the model may need to address one, so a chosen label
    # keeps the typeable key beside it (``Platform Reliability (ops)``) while
    # a derived default paints whatever the human surfaces paint.
    row = (
        f"- {identifier_text(context, display_form(team.name, team.label))} "
        f"[{slots} {member_word}, led by {identifier_text(context, team.manager)}]: {summary}"
    )
    return truncate_row(row, cap=_ROW_CAP)


async def _op_list(context: ToolContext | None, tool_call_id: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "team", "no team registry attached to this session.")
    try:
        teams = registry.list_teams()
    except (TeamRegistryLockTimeout, TeamRegistryRecoveryError) as exc:
        return _error(tool_call_id, "team", str(exc))
    if not teams:
        body = (
            "no teams registered. Create one with op='create' after agreeing "
            "the manager, the roster, how they collaborate, and (if this "
            "instance owns a product) the project brief. Read guide://teams first."
        )
        return _text(tool_call_id, "team", body)
    body = "registered teams:\n" + "\n".join(_row(team, context=context) for team in teams)
    body += "\n\nlaunch with /team <name> <request>, or op='show' name='<name>'."
    text, spill = spill_truncate(body, "team", context)
    return _text(tool_call_id, "team", text, details=spill or None)


async def _op_show(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "team", "no team registry attached to this session.")
    try:
        team = registry.get_team_by_name(name)
    except (TeamRegistryLockTimeout, TeamRegistryRecoveryError) as exc:
        return _error(tool_call_id, "team", str(exc))
    if team is None:
        return _error(
            tool_call_id,
            "team",
            f"no team named {identifier_text(context, repr(name))} (try op='list')",
        )
    header = [
        f"{identifier_text(context, display_form(team.name, team.label))}. "
        f"Led by {identifier_text(context, team.manager)}",
        f"description: {team.description or '(unstated)'}",
    ]
    if team.aliases:
        # D5: an alias resolves to this team, and the model has no other way
        # to learn one exists. Identifier fields: every alias is a name.
        header.append(
            f"aliases: {', '.join(identifier_text(context, alias) for alias in team.aliases)}"
        )  # i18n: ignore model-facing tool output
    header.append("roster:")  # i18n: ignore model-facing tool output
    header.extend(team.roster_lines(lambda value: identifier_text(context, value)))
    header.extend(_override_lines(team, lambda value: identifier_text(context, value)))
    body = "\n".join(header)
    if team.instructions.strip():
        body += "\n\ncollaboration:\n" + team.instructions.strip()
    else:
        body += "\n\ncollaboration: (none)"
    if team.project.strip():
        body += "\n\nproject:\n" + team.project.strip()
    else:
        body += "\n\nproject: (none)"
    body += f"\n\nlaunch with /team {identifier_text(context, team.name)} <request>."
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
    try:
        existing = registry.get_team_by_name(name)
    except (TeamRegistryLockTimeout, TeamRegistryRecoveryError) as exc:
        return _error(tool_call_id, "team", str(exc))
    from local_operator.profile_scope import (
        durable_write_origin,
        durable_write_refusal,
        row_scope,
        scope_store,
        scope_words,
    )

    scope = params.scope or "session"
    # WHERE THIS WRITE LANDS: a create goes where the caller asked, an update to
    # the row's owner. The consent question is asked for the ACCOUNT store only,
    # and only for a tool call — the TUI/desktop hand a plain registry and no
    # context because the user is the actor there.
    target = row_scope(registry, existing) if existing is not None else scope
    authorized_by: str | None = None
    if context is not None and target == "global":
        intent = DurableWriteIntent(
            kind="team", operation="update" if not creating else "create", name=name
        )
        authorized_by = await durable_write_origin(context, intent)
        if authorized_by is None:
            return _error(tool_call_id, "team", durable_write_refusal(intent))
    if creating and existing is not None:
        # Q1: a create whose name collides with an ALIAS must say so in the
        # registry's own words -- "already exists" sends the model looking
        # for a team by that name that it will never find. The registry makes
        # the same distinction under the writer lock; this pre-check just
        # says it earlier, with the same wording.
        if existing.name.casefold() != name.casefold():
            return _error(
                tool_call_id,
                "team",
                f"Team name {identifier_text(context, repr(name))} is already an alias "
                f"of team {identifier_text(context, repr(existing.name))}",
            )
        return _error(
            tool_call_id,
            "team",
            f"team {identifier_text(context, repr(name))} already exists; "
            "use op='update' to change it.",
        )
    if not creating and existing is None:
        return _error(
            tool_call_id,
            "team",
            f"no team named {identifier_text(context, repr(name))} to update.",
        )
    if params.overrides:
        # The operator gate (design §6): a roster-level pin is the operator's
        # choice unless they have delegated model choice, exactly like a role's
        # own ``effort`` tag. Evaluated against the LIVE config, so a setting
        # flipped back to ``operator`` is obeyed by the very next call.
        from local_operator.harness.subagent import model_may_choose_tier

        if not model_may_choose_tier():
            return _error(tool_call_id, "team", _overrides_refusal())
    try:
        members = parse_members(params.members) if params.members is not None else None
        if members is not None:
            # A ``members=`` update replaces the roster wholesale, so anything
            # the operator had pinned on a surviving slot rides forward unless
            # this call sets it (``_carry_overrides``).
            members = _carry_overrides(existing, members)
        if params.overrides:
            members = _apply_overrides(
                members if members is not None else list(existing.members if existing else []),
                params.overrides,
            )
        instructions = _bounded_field(params.instructions, "collaboration instructions")
        project = _bounded_field(params.project, "project instructions")
    except ValidationError as exc:
        # A word outside the vocabulary is the model's own call to correct, so it
        # gets the per-field ``invalid arguments:`` refusal rather than a bare
        # sentence (and stays a model fault, not an environmental one). Raised by
        # ``_apply_overrides``, which validates the merged slot through the model.
        return _validation_error(tool_call_id, "team", exc)
    except ValueError as exc:
        return _error(tool_call_id, "team", str(exc))

    fields = TeamEditFields(
        name=name if creating else None,
        description=params.description,
        manager=params.manager,
        members=members,
        label=params.label,
        aliases=params.aliases,
        instructions=instructions,
        project=project,
    )
    try:
        if creating:
            if not (params.manager or "").strip():
                fields.manager = "manager"
            team = scope_store(registry, scope).create_team(fields)
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
    pinned = sum(1 for member in team.members if not member.is_default())
    # Identifier fields: the write receipt names the team and its manager; both
    # are marked spans, never credentials.
    note = ""
    unconfigured = _unconfigured_tiers(team.members)
    if unconfigured:
        # Said HERE rather than refused: config moves independently of teams, so
        # a write must not fail for a reason outside the team -- but the operator
        # should learn that the slot will refuse to launch while the edit is
        # still on screen, not on the launch.
        note = (
            f" No model is configured for tier {', '.join(repr(t) for t in unconfigured)} "
            "in values.subagents.models, so a slot pinned to it will refuse to launch."
        )
    return _text(
        tool_call_id,
        "team",
        f"{verb} team {identifier_text(context, repr(team.name))} in {scope_words(target)}. "
        f"{identifier_text(context, team.manager)} leads it; "
        f"{len(team.members)} member slot(s)."
        + (f" {pinned} carries a model/effort override." if pinned else "")
        + note
        + f" Launch with /team {identifier_text(context, team.name)} <request>.",
        # The team receipt used to carry no details at all, so a transcript audit
        # could not tell where a team was written. Same fields as the agent
        # receipt: the store, and the consent origin for an account write.
        details={
            "name": team.name,
            "scope": target,
            "authorized_by": authorized_by,
        },
    )


def _unconfigured_tiers(members: Iterable[TeamMember]) -> list[str]:
    """The override tiers this config has no model for, sorted and deduplicated.

    Read through :func:`configured_effort_tiers` -- the same reader the ``task``
    schema uses to decide which tiers to advertise -- so a tier the tool says is
    unconfigured is exactly one the launch will refuse.
    """
    wanted = sorted({str(member.tier) for member in members if member.tier})
    if not wanted:
        return []
    from local_operator.harness.subagent import configured_effort_tiers

    configured = configured_effort_tiers()
    return [tier for tier in wanted if not configured.get(tier)]


async def _op_delete(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "team_delete", "no team registry attached to this session.")
    try:
        team = registry.get_team_by_name(name)
    except (TeamRegistryLockTimeout, TeamRegistryRecoveryError) as exc:
        return _error(tool_call_id, "team_delete", str(exc))
    if team is None:
        return _error(
            tool_call_id,
            "team_delete",
            f"no team named {identifier_text(context, repr(name))} to delete.",
        )
    from local_operator.profile_scope import row_scope, scope_words

    # RESOLVED SESSION-FIRST, THEN REFUSED IF IT IS THE ACCOUNT'S. The model may
    # remove a team it authored for this conversation; deleting one the user
    # saved for every conversation stays on the user surfaces, which ask their
    # own confirmation. A plain registry (no scopes) reads as global, so a
    # headless tool cannot delete one either.
    target = row_scope(registry, team)
    if target != "session":
        return _error(
            tool_call_id,
            "team_delete",
            f"team {identifier_text(context, repr(name))} is saved in the user's account, "
            "so this tool cannot delete it. Delete it from the desktop Teams editor, the "
            "TUI /team listing, or `lop teams delete`.",
        )
    try:
        registry.delete_team(team.id)
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "team_delete", f"could not delete team: {exc}")
    # Identifier field: the delete receipt names the team and the store it left.
    return _text(
        tool_call_id,
        "team_delete",
        f"deleted team {identifier_text(context, repr(name))} from {scope_words(target)}; "
        "the user's account is unchanged.",
        details={"name": name, "scope": target, "authorized_by": None},
    )


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
