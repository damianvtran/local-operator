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

import asyncio
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
from local_operator.hub_sync.definition import DefinitionRevisionMismatch
from local_operator.profile_scope import (
    durable_write_decision,
    durable_write_refusal,
    load_profile_backup,
    profile_diff_summary,
    save_profile_backup,
    team_revision,
    update_profile_backup,
)
from local_operator.projects import truncate_row
from local_operator.teams import (
    MAX_TEAM_INSTRUCTIONS_CHARS,
    Team,
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
    saver: Literal["on", "off", "inherit"] | None = Field(
        default=None,
        description="Pin Radient Saver for this slot; 'inherit' clears it.",
    )


class TeamParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "create", "update", "promote", "revert"] = Field(
        description=(
            "list/show: what teams exist and what they say; create/update: "
            "author or fix a team; promote: copy THIS conversation's team into "
            "the user's account; revert: undo an account save by its backup_id. "
            "Use team_delete for irreversible removal."
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
            "Default: this conversation if it has a store, else the account. "
            "'global': the account, needs confirmation."
        ),
    )
    # ``as`` is a Python keyword, so the design's ``[as=Y]`` is ``as_name`` here
    # too — one spelling on every surface rather than a translation per layer.
    as_name: str | None = Field(
        default=None, description="promote: copy it under this name (default: its own)."
    )
    backup_id: str | None = Field(
        default=None,
        description="revert: the backup_id an account save's receipt returned.",
    )
    # The flag IS the confirmation, exactly as on the agent twin: one answer, one
    # field, so a caller cannot ask for the overwrite while dropping the part that
    # made it the user's decision.
    confirm_replace: bool = Field(
        default=False,
        description=(
            "promote: replace any account row this would collide with (the team or "
            "a member). This IS the user's confirmation — set it only when they asked."
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
        if member.saver is not None:
            parts.append(
                f"saver {'on' if member.saver else 'off'}"
            )  # i18n: ignore model-facing tool output
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
        for field in ("tier", "effort", "saver"):
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
    # GROUP BY STORE (AC13 / U1): a conversation team and an account team have
    # different lifetimes, and a model choosing a roster must see which it is.
    from local_operator.profile_scope import row_scope

    here = [team for team in teams if row_scope(registry, team) == "session"]
    everywhere = [team for team in teams if row_scope(registry, team) != "session"]
    body = ""
    if here:
        body += "teams in this conversation:\n" + "\n".join(
            _row(team, context=context) for team in here
        )
    if everywhere:
        if body:
            body += "\n\n"
        body += "teams saved everywhere (your account):\n" + "\n".join(
            _row(team, context=context) for team in everywhere
        )
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
            f"no team named {identifier_text(context, repr(name))} in this conversation "
            "or your account (try op='list'). A team can be conversation-local, so one "
            "created in another conversation is not visible here; a fork of this "
            "conversation starts with a copy of its teams.",
        )
    from local_operator.profile_scope import row_scope, scope_words

    header = [
        f"{identifier_text(context, display_form(team.name, team.label))}. "
        f"Led by {identifier_text(context, team.manager)}",
        f"scope: {scope_words(row_scope(registry, team))}",
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
        durable_write_decision,
        durable_write_refusal,
        needs_durable_consent,
        row_scope,
        scope_store,
        scope_words,
        shadow_in_other_store,
        shadow_refusal,
        tool_write_scope,
    )

    scope = tool_write_scope(registry, params.scope)
    # THE COLLISION RUNS BEFORE THE CONSENT ASK (agent review round 1, M1): a
    # create over an account name must answer "that name exists", not ask for an
    # account write the model cannot authorize. Case-folded and ALIAS-aware (the
    # helper), because `/team release` resolves an alias exactly like a name.
    if creating:
        collides = shadow_in_other_store(registry, scope, name, kind="team")
        if collides is not None:
            return _error(
                tool_call_id,
                "team",
                shadow_refusal(
                    "team",
                    name,
                    existing_scope="global" if scope == "session" else "session",
                ),
            )
    # WHERE THIS WRITE LANDS: a create goes where the caller asked, an update to
    # the row's owner. The consent question is asked for the ACCOUNT store only,
    # and only for a tool call — the TUI/desktop hand a plain registry and no
    # context because the user is the actor there.
    target = row_scope(registry, existing) if existing is not None else scope
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
    authorized_by: str | None = None
    #: Set only for an approved ACCOUNT update: the revision the card was shown
    #: against (the write is CAS-bound to it) and the backup taken at it.
    base_revision: str | None = None
    backup_id: str | None = None
    backup_payload: dict[str, Any] = {}
    changes: tuple[str, ...] = ()
    summary = ""
    if not creating and existing is None:
        return _error(
            tool_call_id,
            "team",
            f"no team named {identifier_text(context, repr(name))} to update in this "
            "conversation or your account (a team can be conversation-local; try "
            "op='list').",
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
    if context is not None and target == "global" and needs_durable_consent(registry, params.scope):
        changes, summary = _team_changes(existing, params)
        base_revision = team_revision(existing) if existing is not None else None
        intent = DurableWriteIntent(
            kind="team",
            operation="update" if not creating else "create",
            name=name,
            changes=changes,
            summary=summary,
            base_revision=base_revision,
            proposal=_team_proposal(existing, fields, creating, name),
            tool_call_id=tool_call_id,
        )
        authorized_by, reason = await durable_write_decision(context, intent)
        if reason == "proposed":
            from local_operator.tools.agent_tool import _proposed_profile_receipt

            return _proposed_profile_receipt(tool_call_id, "team", authorized_by)
        if authorized_by is None:
            return _error(tool_call_id, "team", durable_write_refusal(intent, reason))
        if existing is not None:
            # The backup is the row at the APPROVED revision; a row that moved
            # since the card refuses here, and one that moves after refuses at
            # the CAS-bound write below.
            current = registry.get_team(existing.id)
            if team_revision(current) != base_revision:
                return _error(tool_call_id, "team", _moved_refusal(name))
            backup_id = save_profile_backup(
                getattr(context, "session_dir", None),
                {
                    "kind": "team",
                    "name": current.name,
                    "id": str(current.id),
                    "scope": "global",
                    "row": current.model_dump(mode="json"),
                    "base_revision": base_revision,
                    "authorized_by": authorized_by,
                },
                on_saved=backup_payload.update,
            )
    try:
        if creating:
            if not (params.manager or "").strip():
                fields.manager = "manager"
            team = scope_store(registry, scope).create_team(fields)
            verb = "created"
        else:
            assert existing is not None  # guarded above
            team = registry.update_team(
                existing.id,
                fields,
                precondition=(
                    _revision_precondition(base_revision) if base_revision is not None else None
                ),
            )
            verb = "updated"
    except DefinitionRevisionMismatch:
        return _error(tool_call_id, "team", _moved_refusal(name))
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
    # The team receipt used to carry no details at all, so a transcript audit
    # could not tell where a team was written. Same fields as the agent
    # receipt: the store, and the consent origin for an account write.
    details: dict[str, Any] = {
        "name": team.name,
        "scope": target,
        "authorized_by": authorized_by,
    }
    receipt = ""
    if authorized_by:
        receipt = _team_receipt(
            context, team, changes, summary, backup_id, details, tool_call_id, backup_payload
        )
    return _text(
        tool_call_id,
        "team",
        f"{verb} team {identifier_text(context, repr(team.name))} in {scope_words(target)}. "
        f"{identifier_text(context, team.manager)} leads it; "
        f"{len(team.members)} member slot(s)."
        + (f" {pinned} carries a model/effort override." if pinned else "")
        + note
        + f" Launch with /team {identifier_text(context, team.name)} <request>."
        + receipt,
        details=details,
    )


#: The team fields an account save can change, in the tool's vocabulary.
_TEAM_SAVE_FIELDS = (
    "label",
    "description",
    "manager",
    "members",
    "overrides",
    "aliases",
    "instructions",
    "project",
)


def _team_proposal(
    existing: Any, fields: TeamEditFields, creating: bool, name: str
) -> dict[str, Any]:
    import uuid

    from local_operator.display_labels import default_label
    from local_operator.profile_approval import TeamImage

    before = (
        None
        if existing is None
        else {key: existing.model_dump(mode="json")[key] for key in TeamImage.model_fields}
    )
    after = (
        dict(before)
        if before is not None
        else {
            "id": str(uuid.uuid4()),
            "name": name,
            "label": default_label(name),
            "description": "",
            "manager": "manager",
            "members": [],
            "aliases": [],
            "instructions": "",
            "project": "",
        }
    )
    after.update(fields.model_dump(mode="json", exclude_none=True))
    if not after["label"]:
        after["label"] = default_label(name)
    return {
        "profile_kind": "team",
        "operation": "create" if creating else "update",
        "target_id": after["id"],
        "name": name,
        "before": before,
        "after": after,
    }


def _team_changes(existing: Any, params: TeamParams) -> tuple[tuple[str, ...], str]:
    """The fields this call names, and a short diff summary for the card/receipt.

    Members and overrides are where model pins live on a team, so a change to
    either is called out as a model/effort change rather than a bare field name.
    """
    from local_operator.tools.agent_tool import _format_field, _instruction_diff

    changes = tuple(
        field for field in _TEAM_SAVE_FIELDS if getattr(params, field, None) is not None
    )
    lines: list[str] = []
    for field in changes:
        new = getattr(params, field)
        old = getattr(existing, field, None) if existing is not None else None
        if field == "instructions":
            rendered, is_diff = _instruction_diff(
                str(old or ""), str(new or ""), fromfile="before", tofile="after"
            )
            heading = "instructions" if is_diff else "instructions (replaced; new text)"
            lines.append(f"{heading}:\n{rendered}")
        elif field in ("members", "overrides"):
            lines.append(  # i18n: ignore Model-facing definition diff; policy fields stay English
                f"{field} (roster and model/effort pins): replaced as given"
            )
        else:
            lines.append(f"{field}: {_format_field(old)} -> {_format_field(new)}")
    return changes, "\n".join(lines)


def _moved_refusal(name: str) -> str:
    return (
        f"team {name!r} changed after this save was approved, so nothing was written. "
        "Show the user the current team and propose the change again."
    )


def _revision_precondition(expected: str) -> Callable[[Any], None]:
    """Refuse under the team registry's lock when the row left ``expected``."""

    def check(current: Any) -> None:
        if team_revision(current) != expected:
            raise DefinitionRevisionMismatch("the team changed since it was approved")

    return check


def _team_receipt(
    context: ToolContext | None,
    team: Any,
    changes: tuple[str, ...],
    summary: str,
    backup_id: str | None,
    details: dict[str, Any],
    tool_call_id: str = "",
    backup_payload: dict[str, Any] | None = None,
) -> str:
    """The account-save receipt: fields, diff, backup and the exact revert call."""
    revision = team_revision(team)
    details.update(
        fields=list(changes), diff=summary.splitlines(), revision=revision, backup_id=backup_id
    )
    changed = "\n" + profile_diff_summary(summary)
    if not backup_id:
        return changed
    update_profile_backup(getattr(context, "session_dir", None), backup_id, saved_revision=revision)
    details["revert"] = {"op": "revert", "backup_id": backup_id}
    register = getattr(context, "register_profile_action", None)
    if register is not None and backup_payload and context is not None and context.session_dir:
        from pathlib import Path

        from local_operator.definition_transaction import commitment
        from local_operator.profile_approval import TeamImage

        before = {key: backup_payload["row"][key] for key in TeamImage.model_fields}
        after = {key: team.model_dump(mode="json")[key] for key in TeamImage.model_fields}
        root = Path(getattr(context, "team_registry").config_dir)
        details["profile_action"] = register(
            root=root,
            profile_kind="team",
            profile_name=team.name,
            target_id=str(team.id),
            tool_call_id=tool_call_id,
            receipt_id=backup_id,
            before=before,
            after=after,
            backup_path=Path(context.session_dir) / "profile-backups" / (backup_id + ".json"),
            backup_commitment=commitment(dict(backup_payload, saved_revision=revision)),
        )

    return f"{changed}\nRevert with team(op='revert', backup_id='{backup_id}')."


async def _op_revert(context: ToolContext | None, tool_call_id: str, backup_id: str) -> ToolResult:
    """Restore an account team from the backup a granted save took.

    Same rules as the agent twin: the row must still be at the revision
    recorded right after that save (a later edit is never clobbered), and the
    restore goes through the same consent seam as the save.
    """
    registry = _registry(context)
    record = load_profile_backup(
        getattr(context, "session_dir", None) if context is not None else None, backup_id
    )
    if registry is None or record is None or record.get("kind") != "team":
        return _error(tool_call_id, "team", f"no team backup {backup_id!r} in this conversation.")
    name = str(record.get("name") or "")
    saved = record.get("saved_revision")
    try:
        current = registry.get_team(str(record.get("id") or ""))
    except Exception:  # noqa: BLE001 — a deleted team cannot be reverted in place
        current = None
    if current is None:
        return _error(tool_call_id, "team", f"team {name!r} no longer exists; nothing to revert.")
    if current.name != name:
        return _error(
            tool_call_id,
            "team",
            "backup identity does not match the target team; nothing was written.",
        )
    if not saved or team_revision(current) != saved:
        return _error(
            tool_call_id,
            "team",
            f"team {name!r} was edited after that save, so reverting would undo the later "
            "change too; nothing was written. Show the user the current team instead.",
        )
    row = Team.model_validate(record.get("row") or {})
    if str(row.id) != str(current.id) or row.name != current.name:
        return _error(
            tool_call_id,
            "team",
            "backup identity does not match the target team; nothing was written.",
        )
    changed = tuple(
        key
        for key in _TEAM_SAVE_FIELDS
        if key != "overrides" and getattr(row, key, None) != getattr(current, key, None)
    )
    proposed = TeamParams(
        op="update",
        name=name,
        **{key: getattr(row, key) for key in changed if key != "members"},
    )
    changes, summary = _team_changes(current, proposed)
    if "members" in changed:
        changes = (*changes, "members")
        summary += "\nmembers (roster and model/effort pins): restored from backup"
    from local_operator.profile_approval import TeamImage

    intent = DurableWriteIntent(
        kind="team",
        operation="revert",
        name=current.name,
        changes=changes,
        summary=summary,
        base_revision=str(saved),
        proposal={
            "profile_kind": "team",
            "operation": "revert",
            "target_id": str(current.id),
            "name": name,
            "before": {key: current.model_dump(mode="json")[key] for key in TeamImage.model_fields},
            "after": {key: row.model_dump(mode="json")[key] for key in TeamImage.model_fields},
        },
        tool_call_id=tool_call_id,
    )
    authorized_by, reason = await durable_write_decision(context, intent)
    if reason == "proposed":
        from local_operator.tools.agent_tool import _proposed_profile_receipt

        return _proposed_profile_receipt(tool_call_id, "team", authorized_by)
    if authorized_by is None:
        return _error(tool_call_id, "team", durable_write_refusal(intent, reason))
    from pathlib import Path

    from local_operator.profile_actions import restore_bound_profile

    try:
        restore_bound_profile(
            Path(registry.config_dir),
            "team",
            str(current.id),
            {key: current.model_dump(mode="json")[key] for key in TeamImage.model_fields},
            {key: row.model_dump(mode="json")[key] for key in TeamImage.model_fields},
        )
    except DefinitionRevisionMismatch:
        return _error(
            tool_call_id, "team", f"team {name!r} was edited after that save; nothing was written."
        )
    return _text(
        tool_call_id,
        "team",
        f"reverted team {identifier_text(context, repr(name))} in your account to its "
        f"state before backup {backup_id}.",
        details={
            "name": name,
            "scope": "global",
            "authorized_by": authorized_by,
            "reverted_backup_id": backup_id,
            "fields": list(changes),
            "diff": summary.splitlines(),
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
            f"no team named {identifier_text(context, repr(name))} to delete in this "
            "conversation or your account (a team can be conversation-local; try "
            "op='list').",
        )
    from local_operator.profile_scope import row_scope, scope_words

    # RESOLVED SESSION-FIRST, THEN REFUSED IF IT IS THE ACCOUNT'S. The model may
    # remove a team it authored for this conversation; deleting one the user
    # saved for every conversation stays on the user surfaces, which ask their
    # own confirmation. A plain registry (no scopes) reads as global, so a
    # headless tool cannot delete one either.
    target = row_scope(registry, team)
    # A host with NO conversation store keeps pre-S2 behaviour: its plain
    # registry IS the account, and team deletion was a working tool there. The
    # account-row refusal only protects a SCOPED host, where the row the user
    # saved for every conversation is not the model's to remove.
    if target != "session" and getattr(registry, "session_registry", None) is not None:
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
    if params.op == "promote":
        return await _op_promote(context, tool_call_id, params)
    if params.op == "revert":
        if not (params.backup_id or "").strip():
            return _error(tool_call_id, "team", "op='revert' needs 'backup_id'.")
        return await _op_revert(context, tool_call_id, str(params.backup_id).strip())
    return await _op_write(context, tool_call_id, params, creating=params.op == "create")


async def _op_promote(
    context: ToolContext | None, tool_call_id: str, params: TeamParams
) -> ToolResult:
    """Copy a CONVERSATION team (and the members it needs) into the account.

    Same rule as the agent twin (:func:`agent_tool._op_promote`): the copy is the
    user's decision, so a tool call with no host grant is REFUSED with a remedy
    naming the surfaces that can ask. The expansion is the composite's job — it
    resolves the roster through the launch resolver and copies session-scoped
    members in the same transaction — so this op only names the intent and
    reports what came back.
    """

    from local_operator.profile_scope import (
        PromotionIncomplete,
        durable_write_decision,
        durable_write_refusal,
        promote_missing,
    )

    registry = _registry(context)
    agents = getattr(context, "agent_registry", None) if context else None
    if registry is None or getattr(registry, "session_registry", None) is None:
        return _error(
            tool_call_id,
            "team",
            "this run has no conversation store, so there is no conversation-scoped team "
            "to promote. The user can author one on their own surface instead.",
        )
    if agents is None:
        return _error(
            tool_call_id,
            "team",
            "promoting a team needs this conversation's agent registry: its members must "
            "resolve, and conversation-scoped ones are copied with it.",
        )
    name = str(params.name or "").strip()
    intent = DurableWriteIntent(kind="team", operation="promote", name=name)
    authorized_by, reason = await durable_write_decision(context, intent)
    if authorized_by is None:
        return _error(tool_call_id, "team", durable_write_refusal(intent, reason))
    try:
        result = await asyncio.to_thread(
            lambda: registry.promote(
                name,
                as_name=params.as_name,
                replace=params.confirm_replace,
                authorized_by=authorized_by,
                agents=agents,
            )
        )
    except KeyError:
        return _error(tool_call_id, "team", promote_missing("team", name))
    except PromotionIncomplete as error:
        return _error(tool_call_id, "team", str(error))
    except ValueError as error:
        return _error(tool_call_id, "team", str(error))
    target = str(params.as_name or name)
    expanded = ""
    if result.expanded:
        # The receipt NAMES the members: a user who copied "one team" needs to
        # know which roles came with it, because they are now account-wide too.
        expanded = f" It also copied what the roster needs: {', '.join(result.expanded)}."
    replaced = ""
    if result.replaced:
        replaced = f" It overwrote the account's existing {', '.join(result.replaced)}."
    return _text(
        tool_call_id,
        "team",
        f"copied team {identifier_text(context, repr(target))} to your account."
        f"{expanded}{replaced} Every other conversation can use it and this one keeps "
        "its own copy.",
        details={
            "name": target,
            "scope": "global",
            "authorized_by": authorized_by,
            "created": result.created,
            "expanded": result.expanded,
            "replaced": result.replaced,
            "provenance_ids": result.provenance_ids,
        },
    )


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
            " Account edits the user explicitly requested apply directly; otherwise "
            "review the proposed change in Agents → <name>, or use profile_proposal "
            "after the user agrees."
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
