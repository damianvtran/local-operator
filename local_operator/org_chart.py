"""Resolve a team NAME into an in-memory ORG TREE, ready to lay out and draw.

WHY THIS EXISTS
===============

A team is a flat roster (a manager plus member slots). With nesting (a member
slot may itself reference another team — see ``TeamMember.kind == "team"`` in
:mod:`local_operator.teams`) a team becomes an "org": a tree of teams. The
``/team chart`` TUI surface renders that tree, but rendering is two independent
concerns — turning the DATA into a tree (here), and laying that tree out on a
character grid (:mod:`local_operator.tui.org_layout`). This module is the first
half and is deliberately PURE: no Textual import, no I/O of its own beyond the
registry it is handed, so the whole edge-case table (cycles, depth, missing
refs, count multiplicity) is unit-testable in isolation.

The resolver renders the *declared* org — the structure the data model
describes. Whether the delegation RUNTIME that makes a nested team executable
is wired yet is a separate concern (a follow-up); a team-boundary node is
tagged ``(declared)`` by the renderer so the chart never implies a capability
that is not live.

TOTALITY AND SAFETY
===================

``resolve_org`` never raises on bad data and always terminates:

- a **cycle** (team A nests B which nests A) stops at the second visit to a
  team already on the current path — a ``"cycle"`` leaf, no recursion, proven
  by the ``path`` frozenset carried down each branch;
- **depth** is bounded by ``MAX_ORG_DEPTH`` (a ``"depth"`` leaf past the
  limit), so even a mis-detected cycle cannot run away;
- a **missing** referenced team or agent becomes an ``"unresolved"`` node
  rather than a dropped slot, so the gap is visible rather than silent.

These are data properties, so the guard lives HERE (where the data is
resolved), not in the widget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from local_operator.agent_profiles import classify_name
from local_operator.teams import MAX_ORG_DEPTH, Team, TeamRegistry

#: The kinds a node can be. ``team``/``manager`` describe the org structure;
#: ``role``/``specialist``/``seed`` are resolved agent leaves; the last three
#: are annotation nodes the renderer draws as ghosts/markers.
OrgKind = Literal[
    "team",  # a team boundary (has a manager + members)
    "manager",  # the manager agent of the team above it
    "role",  # an agent leaf resolved to the operator's own role
    "specialist",  # an agent leaf resolved to a specialist
    "seed",  # an agent leaf resolved to a packaged seed
    "unresolved",  # a name that matched nothing (missing team or agent)
    "cycle",  # a team already on the current path (stop here)
    "depth",  # depth limit reached (stop here)
]


@dataclass(frozen=True)
class OrgNode:
    """One node of the resolved org tree.

    Frozen because a resolved tree is a value, not a mutable scene graph: the
    layout module reads it and produces positions in a separate structure, so
    nothing downstream needs to mutate a node in place. ``replace`` carries a
    slot's ``count`` onto a resolved sub-team node without copying the subtree.
    """

    label: str  # display name (team name or agent role)
    kind: OrgKind  # what this node IS (see OrgKind)
    count: int = 1  # slot multiplicity (from TeamMember.count)
    detail: str = ""  # e.g. "manager", "no members", "missing team"
    children: tuple["OrgNode", ...] = ()
    team_id: str | None = None  # set on team/manager nodes, for cycle tracking


def _team_detail(team: Team) -> str:
    """A one-line fact for a team boundary node: who leads it, how many slots."""
    slots = team.member_count()
    if slots == 0:
        return "no members"
    word = "member" if slots == 1 else "members"
    return f"led by {team.manager} · {slots} {word}"


def _agent_node(
    name: str,
    agents: Any,
    *,
    count: int = 1,
    role_of_manager: bool = False,
) -> OrgNode:
    """Resolve one agent name to a leaf node.

    Reuses :func:`classify_name` — the SAME classifier ``/agent`` attach and a
    team's manager resolution use — so the chart's kind tag can never drift
    from what an attach would actually pick. A manager is a ``"manager"`` node
    regardless of what its name classifies as; the classification still runs so
    a manager whose profile is missing reads as unresolved (``detail`` carries
    the resolved kind for the detailed zoom tier).
    """

    # minor-3 — belt-and-suspenders totality. ``classify_name`` ultimately
    # calls ``resolve_profile`` → ``load_seed``, whose final ``read_text`` is
    # guarded only for ``OSError``; a non-OSError there would propagate out of
    # ``resolve_org``, which is called SYNCHRONOUSLY from ``_open_org_chart_view``
    # and must never crash a UI surface. The resolver's "never raises" contract
    # is now load-bearing, so a failed classification degrades to a resolved
    # leaf tagged ``unresolved`` (a visible ghost) rather than an exception.
    try:
        kind = classify_name(name, registry=agents)
    except Exception:  # noqa: BLE001 - a UI surface must never crash on classification
        kind = "unresolved"
    if role_of_manager:
        # A manager keeps the "manager" node kind so the renderer marks it, but
        # records the resolved kind in detail so a missing manager still shows
        # as a gap and the detailed tier can name role/specialist/seed.
        detail = "manager" if kind != "unresolved" else "manager · unresolved"
        return OrgNode(name, "manager", count=count, detail=detail)
    return OrgNode(name, kind, count=count)


def _team_node(
    team: Team,
    teams: TeamRegistry,
    agents: Any,
    *,
    path: frozenset[str],
    depth: int,
) -> OrgNode:
    """Resolve a team into a boundary node with its manager and members.

    ``path`` is the set of team ids ALREADY on the branch from the root to
    here; a team whose id is in it is a cycle and stops. ``depth`` guards the
    absolute nesting limit independently, so the tree is bounded even if a
    cycle somehow slipped the path check.
    """

    if depth > MAX_ORG_DEPTH:
        # Past the limit: a visible truncation marker rather than more
        # recursion. Carries the team_id so a reader can still see WHICH team
        # was cut, and the renderer draws it as a "⋯" node.
        return OrgNode(team.name, "depth", detail="depth limit", team_id=team.id)
    if team.id in path:
        # Second visit to a team already on this branch: a cycle. Stop with a
        # leaf — NO recursion — so A→B→A terminates instead of looping.
        return OrgNode(team.name, "cycle", detail="already shown above", team_id=team.id)
    path2 = path | {team.id}
    children: list[OrgNode] = []
    # 1) the manager is the FIRST child of the team boundary, always present
    #    (an empty team still has a manager to render).
    children.append(_agent_node(team.manager, agents, role_of_manager=True))
    # 2) then each roster slot, in authored order.
    for member in team.members:
        if member.kind == "team":
            # The author declared this slot a nested team. The name is looked up
            # as a TEAM first — the author's kind is authoritative, that is the
            # whole point of the discriminator. A missing team is a visible
            # ghost, never silently reinterpreted as an agent of the same name.
            sub = teams.get_team_by_name(member.role)
            if sub is None:
                node = OrgNode(member.role, "unresolved", count=member.count, detail="missing team")
            else:
                node = _team_node(sub, teams, agents, path=path2, depth=depth + 1)
                # Carry the slot's ×N onto the resolved sub-team node without
                # re-resolving the subtree: the renderer decides per zoom tier
                # whether to badge or draw N copies.
                node = replace(node, count=member.count)
        else:
            node = _agent_node(member.role, agents, count=member.count)
        children.append(node)
    return OrgNode(
        team.name,
        "team",
        detail=_team_detail(team),
        children=tuple(children),
        team_id=team.id,
    )


def resolve_org(name: str, *, teams: Any, agents: Any = None) -> OrgNode:
    """Resolve a top-level team NAME into its org tree.

    The subject of ``/team chart`` is a TEAM, so the name is looked up as a
    team first; if none exists it is an ``"unresolved"`` node (``detail`` says
    why), never reinterpreted as an agent — the command charts teams, and a
    name that is not a team has no org to show. ``agents`` is the agent
    registry used to tag leaf kinds; ``None`` is legal (every leaf then
    classifies against the packaged seeds only).
    """

    top = teams.get_team_by_name(name)
    if top is None:
        return OrgNode(name, "unresolved", detail="no such team")
    return _team_node(top, teams, agents, path=frozenset(), depth=0)
