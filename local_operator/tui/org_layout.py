"""Buchheim/Walker tidy-tree layout on a character grid — PURE, no Textual.

WHY A REAL TIDY TREE AND NOT NAIVE CENTERING
============================================

The org chart draws a tree of variable-width boxes and has to (a) center every
parent over its children and (b) pack sibling subtrees without overlap. The
naive approach — "center a parent over the midpoint of its children" — fails
the moment one subtree is wider than the gap between two parents: cousins from
different parents collide (the classic Reingold–Tilford failure). This module
is the Buchheim–Jünger–Leipert linear-time improvement of Reingold–Tilford,
which fixes exactly that with ``apportion``/``move_subtree`` and threaded
contours, in O(n).

The one adaptation from the textbook is variable node WIDTH: a box is as wide
as its label needs, so the required separation between two adjacent subtree
roots is ``w_a/2 + w_b/2 + H_GAP`` (half of each box plus a gutter) in CELL
units, not a uniform 1. That is the whole change; the contour walk is
unmodified, which is why the throwaway prototype in ``scratch/tidytree_proto``
could prove the geometry before this shipped.

Coordinates are FLOATS during the two walks (so cumulative half-widths do not
drift) and are exposed as floats; the renderer rounds to integer cells at paint
time. This module knows nothing about Textual, rich, or zoom tiers — it takes a
tree of ``LayoutInput`` (label-free: just a width and children) and returns a
flat list of ``Placed`` nodes with ``x`` (cell column of the box CENTER) and
``depth`` (level index). Keeping it free of the render types is what makes the
non-overlap and centering invariants unit-testable on random trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

#: Minimum clear cells between two sibling boxes. Matches the prototype.
H_GAP = 2


@dataclass
class LayoutInput:
    """The layout's view of one node: a width and its children, nothing else.

    Deliberately separate from ``OrgNode`` (which carries display semantics):
    the layout only needs geometry, so a caller maps its own tree onto this and
    reads positions back by identity or index. ``key`` is an opaque handle the
    caller uses to correlate a result back to its source node.
    """

    width: int  # cell width of this node's box (>= 1)
    children: list["LayoutInput"] = field(default_factory=list)
    key: object = None

    # -- Buchheim scratch state (reset by ``layout``); never set by callers --
    x: float = 0.0
    y: int = 0
    prelim: float = 0.0
    mod: float = 0.0
    thread: "LayoutInput | None" = None
    ancestor: "LayoutInput | None" = None
    change: float = 0.0
    shift: float = 0.0
    number: int = 0
    parent: "LayoutInput | None" = None

    def __post_init__(self) -> None:
        # Each node starts as its own ancestor — the contour walk overwrites
        # this as it threads subtrees together.
        self.ancestor = self


@dataclass(frozen=True)
class Placed:
    """One laid-out node: its source ``key``, box center ``x``, and ``depth``."""

    key: object
    x: float  # cell column of the box CENTER (float; renderer rounds)
    width: int
    depth: int


def _sep(a: LayoutInput, b: LayoutInput) -> float:
    # Distance required between the CENTERS of two adjacent subtree roots so
    # their boxes clear by H_GAP: half of each width plus the gap. This is the
    # single line that generalizes the textbook's unit separation to variable
    # widths — everything else is the unmodified contour walk.
    return a.width / 2 + b.width / 2 + H_GAP


def _left_sibling(v: LayoutInput) -> LayoutInput | None:
    if v.parent is None:
        return None
    i = v.parent.children.index(v)
    return v.parent.children[i - 1] if i > 0 else None


def _next_left(v: LayoutInput) -> LayoutInput | None:
    return v.children[0] if v.children else v.thread


def _next_right(v: LayoutInput) -> LayoutInput | None:
    return v.children[-1] if v.children else v.thread


def _first_walk(v: LayoutInput) -> None:
    w_left = _left_sibling(v)
    if not v.children:
        v.prelim = (w_left.prelim + _sep(w_left, v)) if w_left is not None else 0.0
        return
    default_ancestor = v.children[0]
    for w in v.children:
        _first_walk(w)
        default_ancestor = _apportion(w, default_ancestor)
    _execute_shifts(v)
    midpoint = (v.children[0].prelim + v.children[-1].prelim) / 2
    if w_left is not None:
        v.prelim = w_left.prelim + _sep(w_left, v)
        v.mod = v.prelim - midpoint
    else:
        v.prelim = midpoint


def _apportion(v: LayoutInput, default_ancestor: LayoutInput) -> LayoutInput:
    w = _left_sibling(v)
    if w is None:
        return default_ancestor
    # The four contour cursors walk two subtrees in lockstep (inner/outer,
    # left/right). Annotated LayoutInput — the loop guard proves the inner
    # cursors are non-None before each step, and Buchheim's contract keeps the
    # outer cursors advancing in lockstep with them — so the `_next_*` returns
    # (typed Optional) are narrowed by the invariant, not by a runtime check.
    vip: LayoutInput = v
    vop: LayoutInput = v
    vim: LayoutInput = w
    assert vip.parent is not None
    vom: LayoutInput = vip.parent.children[0]
    sip = vip.mod
    sop = vop.mod
    sim = vim.mod
    som = vom.mod
    while _next_right(vim) is not None and _next_left(vip) is not None:
        vim = _next_right(vim)  # type: ignore[assignment]
        vip = _next_left(vip)  # type: ignore[assignment]
        vom = _next_left(vom)  # type: ignore[assignment]
        vop = _next_right(vop)  # type: ignore[assignment]
        vop.ancestor = v
        shift = (vim.prelim + sim) - (vip.prelim + sip) + _sep(vim, vip)
        if shift > 0:
            _move_subtree(_ancestor(vim, v, default_ancestor), v, shift)
            sip += shift
            sop += shift
        sim += vim.mod
        sip += vip.mod
        som += vom.mod
        sop += vop.mod
    if _next_right(vim) is not None and _next_right(vop) is None:
        vop.thread = _next_right(vim)
        vop.mod += sim - sop
    if _next_left(vip) is not None and _next_left(vom) is None:
        vom.thread = _next_left(vip)
        vom.mod += sip - som
        default_ancestor = v
    return default_ancestor


def _move_subtree(wm: LayoutInput, wp: LayoutInput, shift: float) -> None:
    subtrees = wp.number - wm.number
    wp.change -= shift / subtrees
    wp.shift += shift
    wm.change += shift / subtrees
    wp.prelim += shift
    wp.mod += shift


def _execute_shifts(v: LayoutInput) -> None:
    shift = 0.0
    change = 0.0
    for w in reversed(v.children):
        w.prelim += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change


def _ancestor(vim: LayoutInput, v: LayoutInput, default_ancestor: LayoutInput) -> LayoutInput:
    assert v.parent is not None
    if vim.ancestor in v.parent.children:
        return vim.ancestor  # type: ignore[return-value]
    return default_ancestor


def _second_walk(v: LayoutInput, m: float, depth: int, out: list[Placed]) -> None:
    v.x = v.prelim + m
    v.y = depth
    out.append(Placed(key=v.key, x=v.x, width=v.width, depth=depth))
    for w in v.children:
        _second_walk(w, m + v.mod, depth + 1, out)


def _annotate(v: LayoutInput, parent: LayoutInput | None, number: int) -> None:
    # Reset every scratch field so ``layout`` is idempotent — a caller may lay
    # the same tree out again after a zoom change with different widths.
    v.parent = parent
    v.number = number
    v.ancestor = v
    v.prelim = 0.0
    v.mod = 0.0
    v.thread = None
    v.change = 0.0
    v.shift = 0.0
    for i, c in enumerate(v.children):
        _annotate(c, v, i)


def layout(root: LayoutInput) -> list[Placed]:
    """Lay a tree out and return its nodes positioned, root's ``x`` normalized.

    The returned list is in pre-order (root first). ``x`` is the box CENTER in
    cell columns; the leftmost box's left edge is not guaranteed to be zero —
    the renderer computes the grid's min/max x itself so it can allocate the
    canvas. Positions are floats; round at paint time.
    """

    _annotate(root, None, 0)
    _first_walk(root)
    out: list[Placed] = []
    # Shift so the root sits at x=0; the renderer re-bases against the true
    # min-x of all boxes (a wide left subtree can push a box left of the root).
    _second_walk(root, -root.prelim, 0, out)
    return out


def bounds(placed: Sequence[Placed]) -> tuple[float, float, int]:
    """Return ``(min_left, max_right, max_depth)`` over placed boxes.

    Left/right are box EDGES (center ∓ half width), which is what the renderer
    needs to size the canvas; ``max_depth`` is the deepest level index.
    """

    min_left = min(p.x - p.width / 2 for p in placed)
    max_right = max(p.x + p.width / 2 for p in placed)
    max_depth = max(p.depth for p in placed)
    return (min_left, max_right, max_depth)
