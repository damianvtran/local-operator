"""Pure draft/diff model for operator-authored profiles (LOH-19 P3a).

The form never invents persistence semantics: AgentParams and write_profile are
also the tool/desktop boundary. In particular omission means preserve, while an
empty tool list and the explicit inherit sentinel clear role policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Collection


@dataclass(frozen=True)
class AgentDraft:
    name: str = ""
    kind: str = "role"
    label: str = ""
    description: str = ""
    instructions: str = ""
    tools: str = ""
    effort: str = "inherit"
    delegate: bool = False
    action_class: str = "reactive"


@dataclass(frozen=True)
class AgentPlan:
    changes: dict[str, Any]
    error_field: str = ""
    error_code: str = ""


def parse_tools(text: str) -> list[str]:
    """Reject empty segments/whitespace inside names rather than widening access."""
    if not text.strip():
        return []
    names = [part.strip() for part in text.split(",")]
    if any(not name or any(c.isspace() for c in name) for name in names):
        raise ValueError("tools")
    return list(dict.fromkeys(names))


def build_agent_plan(
    initial: AgentDraft | None, draft: AgentDraft, tiers: Collection[str]
) -> AgentPlan:
    """Validate without a message pump; updates contain only changed fields."""

    def refuse(field: str, code: str) -> AgentPlan:
        return AgentPlan({}, field, code)

    if not draft.name.strip() or any(c.isspace() or c in "/\\" for c in draft.name):
        return refuse("name", "invalid_name")
    if not draft.instructions.strip():
        return refuse("instructions", "required_instructions")
    if initial is not None and initial.description.strip() and not draft.description.strip():
        return refuse("description", "description_clear_unsupported")
    try:
        tools = parse_tools(draft.tools)
    except ValueError:
        return refuse("tools", "invalid_tools")
    if draft.kind == "specialist" and (tools or draft.effort != "inherit" or draft.delegate):
        return refuse("kind", "specialist_fields")
    if draft.effort != "inherit" and draft.effort not in tiers:
        return refuse("effort", "invalid_effort")
    changes: dict[str, Any] = {}
    for field in fields(draft):
        key = field.name
        if initial is not None and key in ("name", "kind"):
            continue
        if draft.kind == "specialist" and key in ("tools", "effort", "delegate"):
            continue
        value = getattr(draft, key)
        before = getattr(initial, key) if initial is not None else None
        if key == "tools":
            value = tools
            before = parse_tools(before) if before is not None else None
        elif isinstance(value, str):
            value = value.strip()
            before = before.strip() if isinstance(before, str) else before
        if initial is None or value != before:
            changes[key] = value
    return AgentPlan(changes)
