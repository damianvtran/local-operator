"""Pure team drafts and changed-only plans for the TUI (LOH-19 P3b).

The roster editor deliberately cannot edit policy overrides. Unchanged rosters
are omitted, and changed rosters inherit policy only on surviving identity slots.
Persistence, uniqueness and definition-CAS remain the registry's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Sequence

from local_operator.teams import TeamMember, parse_members, validate_team_name


@dataclass(frozen=True)
class TeamDraft:
    name: str = ""
    label: str = ""
    description: str = ""
    manager: str = "manager"
    members: str = ""
    aliases: str = ""
    instructions: str = ""
    project: str = ""


@dataclass(frozen=True)
class TeamPlan:
    changes: dict[str, Any]
    error_field: str = ""
    error_code: str = ""


def member_text(members: Sequence[TeamMember]) -> str:
    """Render the core grammar, retaining authored slot order and counts."""
    return ", ".join(
        ("team:" if member.kind == "team" else "")
        + member.role
        + (f":{member.count}" if member.count != 1 else "")
        for member in members
    )


def parse_aliases(text: str) -> list[str]:
    if not text.strip():
        return []
    return [validate_team_name(part.strip()) for part in text.split(",")]


def carry_overrides(
    members: Sequence[TeamMember], original: Sequence[TeamMember]
) -> list[TeamMember]:
    previous = {(member.kind, member.role): member for member in original}
    return [
        member.model_copy(
            update={"tier": old.tier, "effort": old.effort} if old is not None else {}
        )
        for member in members
        for old in [previous.get((member.kind, member.role))]
    ]


def build_team_plan(
    initial: TeamDraft | None,
    draft: TeamDraft,
    original_members: Sequence[TeamMember] = (),
) -> TeamPlan:
    def refuse(field: str, code: str) -> TeamPlan:
        return TeamPlan({}, field, code)

    try:
        validate_team_name(draft.name.strip())
    except ValueError:
        return refuse("name", "invalid_name")
    if not draft.manager.strip():
        return refuse("manager", "required_manager")
    if initial is None and not draft.instructions.strip():
        return refuse("instructions", "required_instructions")
    changes: dict[str, Any] = {}
    for field in fields(draft):
        key = field.name
        if initial is not None and key == "name":
            continue
        value = getattr(draft, key)
        before = getattr(initial, key) if initial is not None else None
        # Brief whitespace is content, like the registry's own brief writer.
        if key not in ("instructions", "project"):
            value = value.strip()
            before = before.strip() if isinstance(before, str) else before
        if initial is not None and value == before:
            continue
        try:
            if key == "members":
                if value and any(not part.strip() for part in value.split(",")):
                    raise ValueError("empty member")
                value = carry_overrides(parse_members(value.split(",")), original_members)
            elif key == "aliases":
                value = parse_aliases(value)
        except ValueError:
            return refuse(key, f"invalid_{key}")
        changes[key] = value
    return TeamPlan(changes)
