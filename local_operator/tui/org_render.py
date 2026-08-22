"""Paint a resolved org tree onto a character canvas — PURE, no Textual.

Third of the three org-chart layers: :mod:`local_operator.org_chart` turns a
team name into an :class:`~local_operator.org_chart.OrgNode` tree, this module
turns that tree (at one of three ZOOM tiers) into boxes, hands their widths to
:mod:`local_operator.tui.org_layout` for tidy-tree positioning, and paints the
boxes, connectors, per-team boundary rules, and ghost/cycle/depth markers into
a rich :class:`~rich.text.Text` on a fixed-size grid.

Kept out of the widget (and free of Textual) so the string output and the
geometry are both unit-testable: the widget is just a scroll container around
what ``render_org`` returns, and the invariants the design calls for
(non-overlap, parent-centering, the two-frame settle) are asserted against
``RenderResult.boxes`` here rather than through a screenshot alone.

ZOOM = LEVEL OF DETAIL, not font scaling
========================================

A terminal cannot scale glyphs, so "zoom" re-renders the SAME resolved tree at
a different detail tier:

- **outline (0):** one box per TEAM; agent members collapse into a ``·N`` count
  badge on the team box; nested teams are still drawn as boxes. Fits a big org.
- **standard (1, default):** manager + agent leaves + team boundaries drawn; a
  nested team collapses to a single box (its members hidden) unless expanded.
- **detailed (2):** every level expanded to the resolved depth; boxes name the
  kind (role/specialist/seed); ``count>1`` agents draw as N sibling copies.

``expand_all`` overrides a tier's default collapsing — the widget's
collapse-all/expand-all control (v1 uses whole-canvas expand rather than
per-node focus, which the design lists as the v1 scope; per-node navigation is
a documented follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rich.style import Style
from rich.text import Text

from local_operator.org_chart import OrgNode
from local_operator.tui import org_layout
from local_operator.tui.org_layout import LayoutInput, Placed

Tier = Literal[0, 1, 2]

#: Rows per level band: box row, drop row, elbow-bus row. The child box row is
#: the next band's box row, so a parent's bus sits directly above its children.
LEVEL_H = 3

#: Style keys painted onto the parallel style grid. Resolved to real
#: ``rich.Style`` objects by ``_STYLES`` at paint time so the theme colours can
#: be swapped without touching the grid logic. Bare names, not hexes: the
#: widget passes a resolver so these follow the active theme.
_KIND_STYLE_KEY = {
    "team": "team",
    "manager": "manager",
    "role": "agent",
    "specialist": "agent",
    "seed": "agent",
    "unresolved": "ghost",
    "cycle": "marker",
    "depth": "marker",
}


@dataclass
class _Box:
    """One drawable box: its text, the source node, and layout scratch key."""

    node: OrgNode
    text: str  # the label INSIDE the box (no brackets)
    style_key: str
    children: list["_Box"] = field(default_factory=list)
    layout_in: LayoutInput | None = None


@dataclass(frozen=True)
class PlacedBox:
    """A painted box's geometry, for tests and the widget's virtual size."""

    label: str
    x: float  # box CENTER column
    width: int  # full box width incl. brackets
    depth: int


@dataclass(frozen=True)
class RenderResult:
    """The painted canvas plus the geometry behind it."""

    text: Text
    width: int
    height: int
    boxes: tuple[PlacedBox, ...]


def _count_suffix(node: OrgNode) -> str:
    """``×N`` for a repeated slot, or ``""``."""
    return f" ×{node.count}" if node.count > 1 else ""


def _member_badge(node: OrgNode) -> str:
    """The ``·N`` member-count badge shown on an outline-tier team box.

    Counts EVERY leaf member — resolved agents AND unresolved/cycle/depth
    markers (minor-2) — but not the manager (structural, always one) and not
    nested sub-teams (the outline draws those as their own boxes). Counting the
    unresolved ones is the fix for the "gap is always visible" invariant the
    ``org_chart`` docstring promises: a team whose only member is a missing
    agent used to badge nothing and read as empty; now it reads ``·1 ?`` so the
    presence — and the fact one is broken — survives the collapse to a box.

    A trailing ``?`` flags that at least one counted member is a ghost
    (unresolved/cycle/depth), so "someone is here but unresolved" is legible
    without expanding. A manager-only org still carries no badge.
    """
    leaf_kinds = ("role", "specialist", "seed", "unresolved", "cycle", "depth")
    members = sum(c.count for c in node.children if c.kind in leaf_kinds)
    has_ghost = any(c.kind in ("unresolved", "cycle", "depth") for c in node.children)
    if not members:
        return ""
    return f" ·{members} ?" if has_ghost else f" ·{members}"


def _leaf_label(node: OrgNode, tier: Tier) -> str:
    """The text inside an agent/marker leaf box at a given tier."""
    if node.kind == "unresolved":
        return f"? {node.label}"
    if node.kind == "cycle":
        # D6 — the detailed tier is where the "(already shown above)" hint earns
        # its keep, so the marker carries its detail there; at coarser tiers the
        # amber ↩ alone says "cycles back" without the extra width.
        if tier == 2 and node.detail:
            return f"↩ {node.label} · {node.detail}"
        return f"↩ {node.label}"
    if node.kind == "depth":
        if tier == 2 and node.detail:
            return f"⋯ {node.label} · {node.detail}"
        return f"⋯ {node.label}"
    label = node.label
    if node.kind == "manager":
        # A leading glyph marks the manager so a reader tells it from a peer
        # member without reading the detail; kept ASCII-adjacent so it survives
        # a terminal without full box-drawing fonts.
        label = f"◆ {label}"
    label += _count_suffix(node)
    if tier == 2 and node.kind in ("role", "specialist", "seed"):
        # Detailed tier names WHAT kind of agent this is, which is the audit the
        # tier exists for. The manager already carries its own marker.
        label += f" ({node.kind})"
    return label


def _build_boxes(node: OrgNode, tier: Tier, *, expand_all: bool, top: bool = False) -> _Box:
    """Turn an OrgNode into the box tree for ``tier``.

    Decides, per tier, whether a node draws its children (expanded) or collapses
    to a single box, and what its label reads as.
    """

    if node.kind != "team":
        # An agent leaf, or a ghost/cycle/depth marker: always a single box.
        return _Box(node=node, text=_leaf_label(node, tier), style_key=_style_key(node))

    # A team boundary node.
    if tier == 0:
        # Outline: the team is one box with a member-count badge; its agent
        # members are folded away, but nested teams stay as boxes so the org
        # shape is still legible.
        #
        # D2 — a member badge (``·N``) and a copy badge (``×N``) abutting read
        # as one garbled number (``·1 ×2``). Join them with a ` · ` seam so they
        # read as two distinct facts ("·1 · ×2" = 1 member, 2 copies).
        member = _member_badge(node).strip()  # "·1" / "·1 ?" / ""
        copies = _count_suffix(node).strip()  # "×2" / ""
        badges = " · ".join(part for part in (member, copies) if part)
        label = f"{node.label} {badges}".rstrip() if badges else node.label
        box = _Box(node=node, text=label, style_key="team")
        for child in node.children:
            if child.kind == "team":
                box.children.append(_build_boxes(child, tier, expand_all=expand_all))
        return box

    # Standard/detailed. A team box labels itself; whether it expands depends on
    # the tier and the expand-all override.
    label = node.label + _count_suffix(node)
    if tier == 2:
        label = f"{label} · {node.detail}" if node.detail else label
    box = _Box(node=node, text=label, style_key="team")
    # Tier 2 (or expand_all) expands every team; tier 1 expands only the TOP
    # team and collapses nested teams to boxes until asked. A collapsed team
    # still shows as a box so the reference is visible.
    expand = top or tier == 2 or expand_all
    if expand:
        for child in node.children:
            child_box = _build_boxes(child, tier, expand_all=expand_all)
            # count>1 agents draw as N sibling copies at the detailed tier; other
            # tiers keep the single badged box.
            if tier == 2 and child.kind in ("role", "specialist", "seed") and child.count > 1:
                for _ in range(child.count):
                    single = OrgNode(child.label, child.kind, count=1, detail=child.detail)
                    box.children.append(
                        _Box(
                            node=single,
                            text=_leaf_label(single, tier),
                            style_key=_style_key(single),
                        )
                    )
            else:
                box.children.append(child_box)
    return box


def _style_key(node: OrgNode) -> str:
    return _KIND_STYLE_KEY.get(node.kind, "agent")


def _box_text(inner: str) -> str:
    """Wrap a label in its bracket chrome. Width is exact — no centering pad."""
    return f"[ {inner} ]"


def _to_layout(box: _Box) -> LayoutInput:
    """Recursively map the box tree onto layout inputs, widths in cells."""
    from rich.cells import cell_len

    width = cell_len(_box_text(box.text))
    # Key the layout node by the box's identity, not the box itself: a _Box is
    # a mutable dataclass (unhashable), and identity is all the reverse lookup
    # needs.
    li = LayoutInput(width=max(width, 3), children=[], key=id(box))
    box.layout_in = li
    for child in box.children:
        li.children.append(_to_layout(child))
    return li


# The style resolver signature: a key like "team"/"agent"/"ghost" → Style.
def _default_style(key: str) -> Style:
    return Style()


def render_org(
    root: OrgNode,
    *,
    tier: Tier = 1,
    expand_all: bool = False,
    style_for=_default_style,
) -> RenderResult:
    """Render a resolved org tree to a painted canvas at ``tier``.

    ``style_for`` maps a style key (``team``/``manager``/``agent``/``ghost``/
    ``marker``/``rule``/``connector``) to a ``rich.Style``; the widget passes a
    theme-aware resolver, tests pass the default (no colour) and assert on
    ``.plain`` and ``.boxes``.
    """

    box_root = _build_boxes(root, tier, expand_all=expand_all, top=True)
    layout_root = _to_layout(box_root)
    placed = org_layout.layout(layout_root)
    min_left, max_right, max_depth = org_layout.bounds(placed)

    # Grid dimensions. One cell of margin each side so a box never touches the
    # canvas edge (which reads as clipped). Height covers every level band, the
    # box row of the last level, AND one extra row beneath it for the
    # deepest-level team boundary rules (D1): a team's rule sits one row under
    # its children, which for the deepest teams falls past the last box row.
    width = int(round(max_right - min_left)) + 2
    height = max_depth * LEVEL_H + 2

    # char grid + parallel style-key grid. Space/'' means "untouched".
    chars = [[" "] * width for _ in range(height)]
    styles: list[list[str]] = [[""] * width for _ in range(height)]

    def put(row: int, col: int, s: str, key: str) -> None:
        for i, ch in enumerate(s):
            c = col + i
            if 0 <= row < height and 0 <= c < width:
                chars[row][c] = ch
                styles[row][c] = key

    # Map a placed layout node back to its box, and record a PlacedBox.
    placed_by_key: dict[object, Placed] = {p.key: p for p in placed}
    out_boxes: list[PlacedBox] = []

    def _placed(box: _Box) -> Placed:
        return placed_by_key[id(box)]

    def paint(box: _Box) -> None:
        p = _placed(box)
        cx = p.x - min_left + 1.0  # +1 for the left margin
        row = p.depth * LEVEL_H
        text = _box_text(box.text)
        left = int(round(cx - p.width / 2))
        put(row, left, text, box.style_key)
        out_boxes.append(PlacedBox(label=box.text, x=cx, width=p.width, depth=p.depth))
        if box.children:
            centers = [_placed(c).x - min_left + 1.0 for c in box.children]
            parent_col = int(round(cx))
            # Drop row: a single '│' from the parent's center.
            put(row + 1, parent_col, "│", "connector")
            # Bus row: horizontal line across the children's span, a junction up
            # to the parent drop and a junction down to each child.
            bus_row = row + 2
            lo = int(round(min(centers)))
            hi = int(round(max(centers)))
            for col in range(lo, hi + 1):
                put(bus_row, col, "─", "connector")
            for c_col_f in centers:
                c_col = int(round(c_col_f))
                # '┬' points down into the child box directly below.
                put(bus_row, c_col, "┬", "connector")
            # The parent junction ('┴' up to the drop) is painted LAST so it wins
            # the shared cell when a child sits directly under the parent.
            if lo == hi:
                # Single child directly below: a straight '│' reads cleaner than
                # a ┬/┴ stack.
                put(bus_row, parent_col, "│", "connector")
            else:
                existing = chars[bus_row][parent_col] if 0 <= bus_row < height else " "
                put(bus_row, parent_col, "┼" if existing == "┬" else "┴", "connector")
            for c in box.children:
                paint(c)

    paint(box_root)

    # D1 — per-team boundary rules. The user's headline ask was "boundaries at
    # each level": on a shared leaf row like
    # ``[◆ mgr-a] [coder] [reviewer] [reviewer] [◆ mgr-b] [scout]`` nothing said
    # the first four belong to pod-a and the last two to pod-b. A faint rule
    # under each team's own children, spanning their x-extent, draws that
    # grouping — two separate spans read as two teams at a glance.
    #
    # Painted as a SEPARATE final pass that fills only BLANK cells, so it never
    # clobbers a box or a connector: under a row of leaf members (the case the
    # rule exists for) the row below is empty and the rule is solid and
    # continuous; where a child is itself a sub-team its drop connector already
    # occupies the centre and the rule simply flows around it. The rule's own
    # tier gate keeps outline (tier 0) uncluttered — grouping there is already
    # one-box-per-team — so rules draw at standard and detailed only.
    def paint_boundaries(box: _Box) -> None:
        if box.node.kind == "team" and box.children:
            child_row = _placed(box.children[0]).depth * LEVEL_H
            rule_row = child_row + 1
            lefts = [_placed(c).x - min_left + 1.0 - _placed(c).width / 2 for c in box.children]
            rights = [_placed(c).x - min_left + 1.0 + _placed(c).width / 2 for c in box.children]
            lo = int(round(min(lefts)))
            hi = int(round(max(rights))) - 1
            for col in range(lo, hi + 1):
                if 0 <= rule_row < height and 0 <= col < width and chars[rule_row][col] == " ":
                    # `rule` is its own style key so the theme can render it
                    # fainter than a connector — it is grouping, not structure.
                    chars[rule_row][col] = "─"
                    styles[rule_row][col] = "rule"
        for c in box.children:
            paint_boundaries(c)

    if tier != 0:
        paint_boundaries(box_root)

    # Assemble the Text row by row, coalescing runs of one style key into one
    # span so the output is compact and the plain text is exactly the grid.
    text = Text(no_wrap=True)
    for r in range(height):
        if r:
            text.append("\n")
        # Trailing blanks are dropped so a row's plain text ends at its content
        # (the canvas is padded by the scroll container, not by trailing spaces
        # that would widen every row to the max). The style grid is walked only
        # up to that trimmed end.
        row_end = width
        while row_end > 0 and chars[r][row_end - 1] == " " and not styles[r][row_end - 1]:
            row_end -= 1
        c = 0
        while c < row_end:
            key = styles[r][c]
            start = c
            while c < row_end and styles[r][c] == key:
                c += 1
            segment = "".join(chars[r][start:c])
            text.append(segment, style=style_for(key) if key else None)
    return RenderResult(text=text, width=width, height=height, boxes=tuple(out_boxes))
