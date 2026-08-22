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
    """The ``·N`` agent-member count shown on an outline-tier team box.

    Counts the team's own AGENT members only (not the manager, which is
    structural and always exactly one, and not nested sub-teams, which the
    outline still draws as their own boxes). A manager-only org therefore
    carries no badge rather than a misleading ``·1`` beside its visible
    sub-team boxes.
    """
    agents = sum(c.count for c in node.children if c.kind in ("role", "specialist", "seed"))
    return f" ·{agents}" if agents else ""


def _leaf_label(node: OrgNode, tier: Tier) -> str:
    """The text inside an agent/marker leaf box at a given tier."""
    if node.kind == "unresolved":
        return f"? {node.label}"
    if node.kind == "cycle":
        return f"↩ {node.label}"
    if node.kind == "depth":
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
        label = node.label + _member_badge(node) + _count_suffix(node)
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
    # canvas edge (which reads as clipped). Height covers every level band plus
    # the box row of the last level.
    width = int(round(max_right - min_left)) + 2
    height = max_depth * LEVEL_H + 1

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
