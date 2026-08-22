"""The org-chart renderer: painted boxes, zoom tiers, and the geometry probes.

The renderer is pure (no Textual), so its string output and the box geometry
behind it are asserted directly here — the §7.3 non-overlap and parent-centering
probes on the PAINTED boxes (as opposed to the raw layout coordinates the
layout test checks), the two-frame settle (rendering twice is identical), and
the tier differences.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.org_chart import resolve_org
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tui.org_layout import H_GAP
from local_operator.tui.org_render import PlacedBox, render_org


@pytest.fixture()
def nested(tmp_path: Path) -> TeamRegistry:
    reg = TeamRegistry(tmp_path)
    reg.create_team(
        TeamEditFields(
            name="pod-a",
            manager="mgr-a",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer", count=2)],
        )
    )
    reg.create_team(
        TeamEditFields(name="pod-b", manager="mgr-b", members=[TeamMember(role="scout")])
    )
    reg.create_team(
        TeamEditFields(
            name="org",
            manager="director",
            members=[
                TeamMember(role="pod-a", kind="team"),
                TeamMember(role="pod-b", kind="team", count=2),
            ],
        )
    )
    return reg


def _boxes_by_depth(result):
    out: dict[int, list[PlacedBox]] = {}
    for b in result.boxes:
        out.setdefault(b.depth, []).append(b)
    return out


def test_painted_boxes_do_not_overlap(nested: TeamRegistry) -> None:
    node = resolve_org("org", teams=nested)
    for tier in (0, 1, 2):
        result = render_org(node, tier=tier)
        for _depth, row in _boxes_by_depth(result).items():
            row = sorted(row, key=lambda b: b.x)
            for a, b in zip(row, row[1:]):
                need = a.width / 2 + b.width / 2 + H_GAP
                assert (b.x - a.x) >= need - 1e-9


def test_two_frame_settle(nested: TeamRegistry) -> None:
    """Rendering the same tree twice is byte-identical — no reflow-after-paint."""
    node = resolve_org("org", teams=nested)
    a = render_org(node, tier=1)
    b = render_org(node, tier=1)
    assert a.text.plain == b.text.plain
    assert (a.width, a.height) == (b.width, b.height)


def test_tiers_differ(nested: TeamRegistry) -> None:
    node = resolve_org("org", teams=nested)
    outline = render_org(node, tier=0)
    standard = render_org(node, tier=1)
    detailed = render_org(node, tier=2)
    # Detailed is the widest (every level expanded, kind tags), outline the
    # narrowest (members folded into badges).
    assert outline.width < detailed.width
    # The detailed tier names agent kinds; the outline tier does not.
    assert "(seed)" in detailed.text.plain
    assert "(seed)" not in outline.text.plain
    # Standard sits between: it shows the top team's members but folds nested
    # teams to boxes.
    assert "director" in standard.text.plain


def test_manager_is_marked(nested: TeamRegistry) -> None:
    node = resolve_org("org", teams=nested)
    result = render_org(node, tier=1)
    # The manager box carries the ◆ marker so it reads apart from a peer.
    assert "◆ director" in result.text.plain


def test_one_agent_team_fits_small(tmp_path: Path) -> None:
    reg = TeamRegistry(tmp_path)
    reg.create_team(TeamEditFields(name="solo", manager="boss", members=[TeamMember(role="coder")]))
    node = resolve_org("solo", teams=reg)
    result = render_org(node, tier=1)
    # A two-node column is narrow — comfortably inside an 80-col terminal.
    assert result.width < 80
    assert "boss" in result.text.plain
    assert "coder" in result.text.plain


def test_ghost_and_cycle_markers(tmp_path: Path) -> None:
    reg = TeamRegistry(tmp_path)
    reg.create_team(
        TeamEditFields(name="a", manager="m", members=[TeamMember(role="b", kind="team")])
    )
    reg.create_team(
        TeamEditFields(
            name="b",
            manager="m",
            members=[
                TeamMember(role="a", kind="team"),
                TeamMember(role="missing", kind="team"),
            ],
        )
    )
    node = resolve_org("a", teams=reg)
    result = render_org(node, tier=2)
    # Cycle back-reference and a missing-team ghost both draw with markers.
    assert "↩ a" in result.text.plain
    assert "? missing" in result.text.plain


def test_count_badge_on_outline(nested: TeamRegistry) -> None:
    node = resolve_org("org", teams=nested)
    outline = render_org(node, tier=0)
    # pod-b is nested ×2 — the outline badges the copy count rather than
    # drawing two subtrees.
    assert "×2" in outline.text.plain


def test_empty_team_renders_manager_only(tmp_path: Path) -> None:
    reg = TeamRegistry(tmp_path)
    reg.create_team(TeamEditFields(name="empty", manager="boss", members=[]))
    node = resolve_org("empty", teams=reg)
    result = render_org(node, tier=1)
    assert "boss" in result.text.plain
    # Exactly one box below the team header (the manager), no crash.
    assert len(result.boxes) == 2
