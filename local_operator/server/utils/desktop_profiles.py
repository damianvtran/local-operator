"""Read-only projections over reusable profile/team authorities, not chat agents.

These helpers need no session or execution host. Names deliberately remain the
runtime's attachment keys; registry IDs describe provenance, never conversation
identity. Keep ordinary conversational agents outside this catalogue.
"""

from __future__ import annotations

from typing import Any

from local_operator.action_class import normalize as normalize_action_class
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
from local_operator.hub_sync.definition import definition_revision


def profile_detail(
    registry: Any, name: str, *, detail: bool = True, already_installed: bool | None = None
) -> dict[str, Any]:
    # Force an actual registry read before the tolerant runtime resolver. A
    # failed registry must not masquerade as a successful packaged fallback.
    complete = getattr(registry, "require_complete_metadata", None)
    if complete is not None:
        complete()
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
    # WHERE THIS ROW LIVES, additively (LOH-9 S2). Derived from the registry when
    # it can answer (the session composite), and "global" for a plain registry —
    # this helper is also used by the account-only routes, where every row is the
    # account's. A packaged seed has no row yet, so it reads global: installing it
    # is still a write to one of the two stores.
    scope = "global"
    scope_of = getattr(registry, "scope_of", None)
    if row is not None and callable(scope_of):
        scope = scope_of(row)
    result: dict[str, Any] = {
        "name": resolved,
        "scope": scope,
        # The RAW display label (never a composed form -- the desktop lane
        # computes that per surface, exactly as the TUI does); it rides the
        # same projection the catalogue and GET responses share.
        "label": metadata.label,
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
        # The class, in the EFFECTIVE spelling (absence reads reactive): the
        # route accepts ``action_class`` on write and ``divergent_fields`` can
        # already report ``class``, so a response that carried neither was a
        # payload a client could not read the control's current value from
        # (QA round 1, Q2). Rides the catalogue too — that path builds from
        # THIS projection with ``detail=False``.
        "action_class": normalize_action_class(metadata.action_class),
        "seed_origin": origin,
        "divergent_fields": [],
        # Opaque revision over the authored editable fields, so an editor can
        # round-trip the exact snapshot it edited. Absent where the row is a
        # builtin seed with no installed local editable row.
        "definition_revision": (
            None
            if kind == "seed" or row is None
            else definition_revision(
                "agent", str(row.id), registry.definition_revision_fields(str(row.id))
            )
        ),
    }
    seed = load_seed(origin) if origin else None
    if seed is not None:
        result["divergent_fields"] = list(seed_divergence(metadata, seed))
    if already_installed is not None:
        # Present ONLY on an install response, and only because the caller asked
        # for it: ``install_seed`` returns ``(profile, already_installed)``
        # (agent_profiles.py:548-560) and the route used to discard the second
        # element, so a client could not tell "I installed it" from "it was
        # already there" — which is exactly what the sidebar's install-all
        # summary has to report ("6 installed, 2 already present", contract
        # §5.6). It is deliberately absent from read responses: a GET is not an
        # install, and a field that means nothing there is one a client will
        # eventually read as if it did.
        result["already_installed"] = already_installed
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


def team_effective(agents: Any, teams: Any, name: str) -> dict[str, Any]:
    """What every slot of ``name`` would launch on, and WHICH layer chose it.

    The read the desktop team editor renders as "effective + source" (design §4):
    for each slot, the model TIER, the MODEL it resolves to and the REASONING
    effort, each with the layer that decided it, so an operator can see why a
    member runs where it does without launching it.

    Reads ``harness.subagent.resolve_member_model`` -- the SAME resolver the
    launch and the resume read -- so a screen and a child cannot disagree; a
    second resolver is exactly how this codebase got schema-vs-launch drift.
    The harness import is inside the function: this module is imported by
    ``server.app``, and the boot cost of pulling the harness for one route is
    paid at the route, not at startup.

    Every refusal is reported as DATA (``available: false`` plus the launch's own
    reason text) rather than raised, and ``available: true`` is verified by
    actually building the spec the launch would build -- an editor showing a
    broken tier is more useful than an editor showing an error page.

    COST: building a spec goes through the model catalogue, which is a LOCAL
    read once warm and can hit the network once on a cold cache (measured
    2026-10-16: 0.06-0.69 s per uncached model, then 0.0000-0.0002 s with the
    network blocked). The route runs it on a worker thread for that reason.
    """
    from local_operator.harness.subagent import resolve_member_model

    team = teams.get_team_by_name(name)
    if team is None:
        raise KeyError(name)
    members: list[dict[str, Any]] = []
    for member in team.members:
        slot_kind = str(member.kind)
        role = str(member.role)
        slot_role: str | None = None
        if slot_kind == "team":
            # A nested-team override lives on THIS slot but applies to the
            # sub-team's manager -- the role this roster actually launches --
            # so the lookup keys on the slot's name while the resolution names
            # the manager, exactly as the launch does it.
            sub = teams.get_team_by_name(role)
            if sub is None:
                members.append(_effective_member(member, None, f"no team named {role!r}"))
                continue
            slot_role = role
            role = str(sub.manager)
        try:
            resolution = resolve_member_model(
                role,
                roster=team,
                slot_kind=slot_kind,
                slot_role=slot_role,
                agent_registry=agents,
            )
        except Exception as exc:  # noqa: BLE001 — reported, never raised
            reason = f"the agent registry could not be read ({exc})"
            members.append(_effective_member(member, None, reason))
            continue
        members.append(_effective_member(member, resolution, ""))
    return {"team": team.name, "members": members}


def _effective_member(member: Any, resolution: Any, reason: str) -> dict[str, Any]:
    """One slot's effective row -- the frozen shape P3 keys on.

    ``reason`` non-empty means the resolution itself failed (an unreadable
    registry, an unknown sub-team); otherwise ``resolution.unavailable`` decides
    availability, and the MODEL is only reported once it has been built, for the
    same reason a launch builds it: a tier that cannot produce a spec is not
    "available".

    ``effort.value`` is the level the child will ACTUALLY run at when a model is
    known to clamp against, and the requested level otherwise -- the frozen §1
    rule: "clamped against the model's ladder only when the model is known;
    otherwise it equals ``requested``". "Not known" covers every case with no
    ladder in hand: the launching session's model (which a server cannot name),
    the ``default`` sentinel, an unavailable tier, and a model whose catalogue
    entry carries no reasoning ladder at all. ``requested`` always keeps the
    word the operator wrote, so the pair is the contract whether or not the value
    could be resolved -- and ``auto`` therefore surfaces as ``auto`` when there
    is no ladder to resolve it against, which is the same answer the editor's own
    ``value ?? requested`` fallback renders.
    """
    if resolution is not None and resolution.unavailable:
        reason = resolution.unavailable
    requested = None if resolution is None else resolution.effort
    # ``model.source`` answers "which layer chose the model". A tier the operator
    # WROTE chose it, even when nothing is configured at it or it points at the
    # session's own model through the sentinel -- so only a slot with no tier at
    # all reports ``session``, the one case where the model really does come from
    # the launching session (design §1's sentinel ruling, applied to every tier).
    has_tier = resolution is not None and resolution.tier is not None
    entry: dict[str, Any] = {
        "role": str(member.role),
        "kind": str(member.kind),
        "tier": {
            "value": None if resolution is None else resolution.tier,
            "source": "none" if resolution is None else resolution.tier_source,
        },
        "model": {"value": None, "source": "tier" if has_tier else "session"},
        "effort": {
            "value": requested,
            "requested": requested,
            "source": "model_default" if resolution is None else resolution.effort_source,
        },
        "available": not reason,
        "reason": reason,
    }
    if reason or resolution is None or resolution.selector is None:
        # ``selector is None`` with no failure means "no tier anywhere": the
        # child inherits the launching session's model, which a SERVER cannot
        # name. That is why ``model.value`` is null here rather than a guess.
        return entry
    selector = resolution.selector
    if _is_sentinel(selector):
        # The sentinel resolves to the session's model AT LAUNCH, and the desktop
        # server is not a session: ``source: "tier"`` says a tier chose it while
        # the value stays null, which is the honest pair. ``effort.value`` is
        # left as ``requested`` by the case above -- there is no ladder to clamp
        # against.
        return entry
    provider, _, model_id = str(selector).partition("/")
    try:
        from local_operator.model.configure import build_model_spec

        spec = build_model_spec(provider, model_id)
    except Exception as exc:  # noqa: BLE001 — a reason, not a crash
        entry["available"] = False
        entry["reason"] = f"{selector!r} could not be resolved ({exc})"
        return entry
    entry["model"] = {"value": f"{provider}/{model_id}", "source": "tier"}
    if requested is None:
        # Nothing was asked for, so the level the model sits on IS the answer.
        entry["effort"]["value"] = spec.reasoning_effort
    elif spec.reasoning_efforts:
        entry["effort"]["value"] = _clamped_effort(spec, requested)
    # Any other model: no ladder to clamp against, so the requested level stands
    # (the value set above is already the right answer).
    return entry


def _is_sentinel(selector: Any) -> bool:
    from local_operator.harness.subagent import is_inherit_tier_sentinel

    return is_inherit_tier_sentinel(selector)


def _clamped_effort(spec: Any, requested: str) -> str | None:
    """``requested`` as a level ``spec`` actually runs at.

    Mirrors ``Session._with_tier_effort`` -- the launch's own clamp -- and is
    pinned equal to it by a test, because a display that clamps differently from
    the launch is the drift this endpoint exists to prevent. ``auto`` is the
    member's "this model's own default", not a rung: it maps to the spec's
    documented default before the clamp, as it does at launch.
    """
    from local_operator.model.effort import resolve_effort_in

    wanted = spec.reasoning_default_effort if requested == "auto" else requested
    clamped = resolve_effort_in(spec.reasoning_efforts, spec.reasoning_default_effort, wanted)
    return clamped if clamped is not None else spec.reasoning_effort


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
