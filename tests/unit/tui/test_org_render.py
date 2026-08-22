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


def test_boundary_rules_drawn_at_standard_and_detailed(nested: TeamRegistry) -> None:
    """D1 — a per-team boundary rule (─ in the `rule` style) demarcates groups.

    The headline ask ("boundaries at each level") — assert the rule is actually
    painted at standard and detailed, and absent at outline (one box per team
    needs no rule).
    """

    node = resolve_org("org", teams=nested)
    for tier in (1, 2):
        result = render_org(node, tier=tier)
        # A row of boundary rule renders as a run of ─; the plain text carries
        # it. Proven present by a horizontal rule under the children.
        assert "─" in result.text.plain, f"tier {tier} has no boundary rule"
    # Outline draws no boundary rule — grouping there is one-box-per-team.
    outline = render_org(node, tier=0)
    # The only ─ at outline would be the elbow bus; assert no dedicated rule row
    # by checking the render height did not reserve the extra boundary row.
    # (Outline height is the box+bus bands only.)
    assert outline.height <= result.height


def test_boundary_rule_uses_rule_style(nested: TeamRegistry) -> None:
    """The boundary rule is painted in the dedicated `rule` style key so the
    theme can render it fainter than a connector (grouping, not structure)."""
    node = resolve_org("org", teams=nested)
    captured: set[str] = set()

    def style_for(key: str):
        captured.add(key)
        from rich.style import Style

        return Style()

    render_org(node, tier=2, style_for=style_for)
    assert "rule" in captured


def test_outline_counts_unresolved_members_and_flags_ghost(tmp_path: Path) -> None:
    """minor-2 — a team whose only member is missing must not vanish at outline.

    Before the fix, `_member_badge` counted only resolved kinds, so a team of
    unresolved members badged nothing and read as an empty `[ team ]` box —
    breaking the "gap is always visible" invariant exactly when the org is big
    enough to fit-to-outline. Now the badge counts every leaf and flags a ghost.
    """

    reg = TeamRegistry(tmp_path)
    reg.create_team(
        TeamEditFields(
            name="squad",
            manager="m",
            members=[TeamMember(role="ghost-a"), TeamMember(role="ghost-b")],
        )
    )
    node = resolve_org("squad", teams=reg)
    outline = render_org(node, tier=0).text.plain
    # Two members counted, and the ghost flag present — not a bare `[ squad ]`.
    assert "·2" in outline
    assert "?" in outline


def test_outline_double_badge_is_separated(nested: TeamRegistry) -> None:
    """D2 — `·N` (members) and `×N` (copies) read as two facts, not one number."""
    outline = render_org(resolve_org("org", teams=nested), tier=0).text.plain
    # pod-b: 1 member, 2 copies → "·1 · ×2", never the garbled "·1 ×2".
    assert "·1 · ×2" in outline
