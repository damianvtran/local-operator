"""The org-chart resolver: from a team name to a total, cycle-safe org tree.

Every §3.3 edge case of the design as a table test. The resolver is PURE and
never raises on bad data, so these assert on the SHAPE it returns (the kind of
each node, the count it carries, the detail it annotates) rather than on an
exception. The cycle case additionally asserts the call TERMINATES — the whole
point of the ``path`` frozenset — via a bounded-time guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.org_chart import OrgNode, resolve_org
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> TeamRegistry:
    return TeamRegistry(tmp_path)


def _kinds(node: OrgNode) -> list[str]:
    """Pre-order list of node kinds, for compact shape assertions."""
    out = [node.kind]
    for child in node.children:
        out.extend(_kinds(child))
    return out


def test_one_agent_team(registry: TeamRegistry) -> None:
    registry.create_team(
        TeamEditFields(name="solo", manager="boss", members=[TeamMember(role="coder")])
    )
    node = resolve_org("solo", teams=registry)
    assert node.kind == "team"
    # manager first, then the one member.
    assert [c.kind for c in node.children] == ["manager", "seed"]
    assert node.children[0].label == "boss"
    assert node.children[1].label == "coder"


def test_empty_team_is_manager_only(registry: TeamRegistry) -> None:
    registry.create_team(TeamEditFields(name="empty", manager="boss", members=[]))
    node = resolve_org("empty", teams=registry)
    assert [c.kind for c in node.children] == ["manager"]
    assert node.detail == "no members"


def test_missing_top_team_is_unresolved(registry: TeamRegistry) -> None:
    node = resolve_org("nope", teams=registry)
    assert node.kind == "unresolved"
    assert node.detail == "no such team"
    assert node.children == ()


def test_nested_org_recurses(registry: TeamRegistry) -> None:
    registry.create_team(
        TeamEditFields(name="pod", manager="lead", members=[TeamMember(role="coder")])
    )
    registry.create_team(
        TeamEditFields(
            name="org",
            manager="director",
            members=[TeamMember(role="pod", kind="team")],
        )
    )
    node = resolve_org("org", teams=registry)
    # org(team) -> [director(manager), pod(team) -> [lead(manager), coder(seed)]]
    assert _kinds(node) == ["team", "manager", "team", "manager", "seed"]
    pod = node.children[1]
    assert pod.kind == "team"
    assert pod.team_id is not None


def test_org_within_org(registry: TeamRegistry) -> None:
    registry.create_team(
        TeamEditFields(name="squad", manager="sl", members=[TeamMember(role="scout")])
    )
    registry.create_team(
        TeamEditFields(name="pod", manager="lead", members=[TeamMember(role="squad", kind="team")])
    )
    registry.create_team(
        TeamEditFields(
            name="org", manager="director", members=[TeamMember(role="pod", kind="team")]
        )
    )
    node = resolve_org("org", teams=registry)
    # Three team boundaries deep.
    assert _kinds(node).count("team") == 3


def test_count_gt_one_agent_carries_count(registry: TeamRegistry) -> None:
    registry.create_team(
        TeamEditFields(name="t", manager="m", members=[TeamMember(role="reviewer", count=2)])
    )
    node = resolve_org("t", teams=registry)
    assert node.children[1].count == 2


def test_count_gt_one_team_carries_count(registry: TeamRegistry) -> None:
    registry.create_team(
        TeamEditFields(name="pod", manager="l", members=[TeamMember(role="coder")])
    )
    registry.create_team(
        TeamEditFields(
            name="org",
            manager="d",
            members=[TeamMember(role="pod", kind="team", count=2)],
        )
    )
    node = resolve_org("org", teams=registry)
    pod = node.children[1]
    assert pod.kind == "team"
    assert pod.count == 2
    # The subtree is resolved once; the count is stamped on the returned node.
    assert [c.kind for c in pod.children] == ["manager", "seed"]


def test_cycle_terminates_in_bounded_time(registry: TeamRegistry) -> None:
    """A→B→A must return a ``cycle`` leaf, not loop forever.

    The bound is the assertion: if the ``path`` guard regressed, this would
    recurse until the stack blew rather than fail the shape check. A real
    registry (not a stub) so the lookup path is the shipped one.
    """

    registry.create_team(
        TeamEditFields(name="a", manager="m", members=[TeamMember(role="b", kind="team")])
    )
    registry.create_team(
        TeamEditFields(name="b", manager="m", members=[TeamMember(role="a", kind="team")])
    )
    node = resolve_org("a", teams=registry)
    # a(team) -> [m, b(team) -> [m, a(cycle)]]
    assert _kinds(node) == ["team", "manager", "team", "manager", "cycle"]
    cycle = node.children[1].children[1]
    assert cycle.kind == "cycle"
    assert cycle.children == ()
    assert cycle.detail == "already shown above"


def test_self_cycle_terminates(registry: TeamRegistry) -> None:
    registry.create_team(
        TeamEditFields(name="loop", manager="m", members=[TeamMember(role="loop", kind="team")])
    )
    node = resolve_org("loop", teams=registry)
    assert node.children[1].kind == "cycle"


def test_depth_limit_truncates(registry: TeamRegistry) -> None:
    """A chain longer than MAX_ORG_DEPTH ends in a ``depth`` node, not runaway.

    Built as a straight chain team0 -> team1 -> ... so the depth guard, not the
    cycle guard, is what stops it (every team id is distinct).
    """

    from local_operator.teams import MAX_ORG_DEPTH

    depth = MAX_ORG_DEPTH + 3
    for i in range(depth):
        members = [TeamMember(role=f"t{i + 1}", kind="team")] if i < depth - 1 else []
        registry.create_team(TeamEditFields(name=f"t{i}", manager="m", members=members))
    node = resolve_org("t0", teams=registry)
    # Walk down the team spine; a "depth" node must appear before the leaf team.
    kinds = _kinds(node)
    assert "depth" in kinds


def test_missing_referenced_team_is_ghost(registry: TeamRegistry) -> None:
    registry.create_team(
        TeamEditFields(name="org", manager="d", members=[TeamMember(role="ghost", kind="team")])
    )
    node = resolve_org("org", teams=registry)
    slot = node.children[1]
    assert slot.kind == "unresolved"
    assert slot.detail == "missing team"


def test_authored_kind_is_authoritative(registry: TeamRegistry) -> None:
    """A name that is BOTH a team and an agent resolves by the slot's kind.

    ``pod`` exists as a team; a slot authored ``kind="agent"`` must resolve as
    an agent (it classifies as unresolved here — no such agent), and a slot
    authored ``kind="team"`` resolves as the team. The author's discriminator
    decides, which is the whole point of the field.
    """

    registry.create_team(
        TeamEditFields(name="pod", manager="l", members=[TeamMember(role="coder")])
    )
    registry.create_team(
        TeamEditFields(
            name="org",
            manager="d",
            members=[
                TeamMember(role="pod", kind="agent"),
                TeamMember(role="pod", kind="team"),
            ],
        )
    )
    node = resolve_org("org", teams=registry)
    agent_slot, team_slot = node.children[1], node.children[2]
    # The agent slot did NOT expand into the team (no such agent → unresolved).
    assert agent_slot.kind == "unresolved"
    assert agent_slot.children == ()
    # The team slot expanded.
    assert team_slot.kind == "team"
    assert [c.kind for c in team_slot.children] == ["manager", "seed"]
