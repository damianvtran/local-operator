"""Read-only projections over reusable profile/team authorities, not chat agents.

These helpers need no session or execution host. Names deliberately remain the
runtime's attachment keys; registry IDs describe provenance, never conversation
identity. Keep ordinary conversational agents outside this catalogue.
"""

from __future__ import annotations

from typing import Any

from local_operator.agent_profiles import (
    is_role,
    is_specialist,
    list_seeds,
    load_seed,
    profile_from_agent,
    resolve_profile_or_specialist,
    seed_divergence,
    seed_origin,
)


def profile_detail(registry: Any, name: str, *, detail: bool = True) -> dict[str, Any]:
    # Force an actual registry read before the tolerant runtime resolver. A
    # failed registry must not masquerade as a successful packaged fallback.
    registered = registry.list_agents()
    kind, profile, instructions, resolved = resolve_profile_or_specialist(name, registry=registry)
    if kind is None:
        raise KeyError(name)
    row = next((row for row in registered if row.name == resolved), None)
    if kind == "specialist":
        if row is None:
            raise KeyError(name)
        metadata = profile_from_agent(registry, row)
    else:
        metadata = profile
    assert metadata is not None
    origin = seed_origin(row) if row is not None else None
    result: dict[str, Any] = {
        "name": resolved,
        "kind": "specialist" if kind == "specialist" else "role",
        "source": "builtin" if kind == "seed" else "installed" if origin else "custom",
        "agent_id": str(row.id) if kind != "seed" and row is not None else None,
        "description": (
            str(row.description or "")
            if kind == "specialist" and row is not None
            else metadata.description
        ),
        "tools": list(metadata.tools) if metadata.tools is not None else None,
        "effort": metadata.effort,
        "delegate": metadata.may_delegate,
        "seed_origin": origin,
        "divergent_fields": [],
    }
    seed = load_seed(origin) if origin else None
    if seed is not None:
        result["divergent_fields"] = list(seed_divergence(metadata, seed))
    if detail:
        result["instructions"] = instructions if kind == "specialist" else metadata.instructions
        if seed is not None:
            result["packaged_instructions"] = seed.instructions
    return result


def profile_catalogue(registry: Any) -> list[dict[str, Any]]:
    names = {
        row.name for row in registry.list_agents() if is_role(row) or is_specialist(row)
    } | set(list_seeds())
    # Case-folded aliases may resolve to the same installed profile. Publish its
    # authoritative name once, while preserving the resolver's precedence.
    resolved = {
        item["name"]: item
        for item in (profile_detail(registry, name, detail=False) for name in sorted(names))
    }
    return sorted(resolved.values(), key=lambda item: (item["name"].casefold(), item["name"]))


def team_catalogue(registry: Any) -> list[dict[str, Any]]:
    return [
        row.model_dump(mode="json", exclude={"instructions", "project"})
        for row in sorted(registry.list_teams(), key=lambda row: (row.name.casefold(), row.name))
    ]


def validate_target(agents: Any, teams: Any, kind: str, name: str) -> str:
    """Validate the complete declared launch graph without installing anything.

    Delegation stays lazy. Validation prevents the first manager prompt from
    claiming a roster that cannot resolve, without spawning that roster here.
    """
    if kind == "agent":
        return str(profile_detail(agents, name, detail=False)["name"])
    from local_operator.teams import MAX_ORG_DEPTH

    def visit(candidate: str, ancestors: tuple[str, ...]) -> str:
        team = teams.get_team_by_name(candidate)
        if team is None:
            raise KeyError(candidate)
        if team.name in ancestors or len(ancestors) >= MAX_ORG_DEPTH:
            raise ValueError("The team roster contains a cycle or exceeds the nesting limit")
        profile_detail(agents, team.manager, detail=False)
        for member in team.members:
            if member.kind == "team":
                visit(member.role, (*ancestors, team.name))
            else:
                profile_detail(agents, member.role, detail=False)
        return team.name

    return visit(name, ())
