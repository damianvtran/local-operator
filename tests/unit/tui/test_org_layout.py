"""Buchheim tidy-tree layout invariants — the checks a screenshot cannot make.

The layout is the one piece whose correctness is a precise geometric property,
not a "looks right": for EVERY pair of nodes on the same level the boxes must
clear by ``H_GAP`` (non-overlap), and every parent's ``x`` must equal the
midpoint of its first and last child (the Buchheim centering invariant). These
are asserted on RANDOM trees so the property holds beyond the hand-drawn cases,
plus the specific shape the prototype used.
"""

from __future__ import annotations

import random

from local_operator.tui.org_layout import H_GAP, LayoutInput, Placed, bounds, layout


def _by_depth(placed):
    out: dict[int, list[Placed]] = {}
    for p in placed:
        out.setdefault(p.depth, []).append(p)
    return out


def _assert_no_overlap(placed) -> None:
    for _depth, row in _by_depth(placed).items():
        row = sorted(row, key=lambda p: p.x)
        for a, b in zip(row, row[1:]):
            need = a.width / 2 + b.width / 2 + H_GAP
            assert (b.x - a.x) >= need - 1e-9, f"overlap: {a.key} / {b.key}"


def _assert_centered(root: LayoutInput, placed) -> None:
    by_key = {p.key: p for p in placed}

    def walk(node: LayoutInput) -> None:
        if node.children:
            first = by_key[node.children[0].key].x
            last = by_key[node.children[-1].key].x
            mid = (first + last) / 2
            assert abs(by_key[node.key].x - mid) < 1e-6
            for c in node.children:
                walk(c)

    walk(root)


def _random_tree(rng: random.Random, depth: int) -> LayoutInput:
    width = rng.randint(3, 20)
    node = LayoutInput(width=width, children=[], key=None)
    node.key = id(node)  # unique handle
    if depth > 0:
        for _ in range(rng.randint(0, 4)):
            node.children.append(_random_tree(rng, depth - 1))
    return node


def test_prototype_shape_is_non_overlapping_and_centered() -> None:
    root = LayoutInput(
        12,
        [
            LayoutInput(
                12,
                [
                    LayoutInput(8, key="coder"),
                    LayoutInput(14, key="reviewer"),
                    LayoutInput(11, key="designer"),
                ],
                key="mgr-a",
            ),
            LayoutInput(
                12,
                [
                    LayoutInput(
                        10,
                        [LayoutInput(8, key="scout"), LayoutInput(8, key="coder2")],
                        key="submgr",
                    )
                ],
                key="mgr-b",
            ),
        ],
        key="director",
    )
    placed = layout(root)
    _assert_no_overlap(placed)
    _assert_centered(root, placed)


def test_single_node_lays_out() -> None:
    root = LayoutInput(5, [], key="solo")
    placed = layout(root)
    assert len(placed) == 1
    assert placed[0].depth == 0


def test_single_child_parent_sits_over_child() -> None:
    child = LayoutInput(6, [], key="c")
    root = LayoutInput(10, [child], key="r")
    placed = layout(root)
    by_key = {p.key: p for p in placed}
    # A single child: parent center == child center.
    assert abs(by_key["r"].x - by_key["c"].x) < 1e-6


def test_random_trees_never_overlap_and_stay_centered() -> None:
    rng = random.Random(1234)
    for _ in range(200):
        root = _random_tree(rng, depth=rng.randint(0, 4))
        placed = layout(root)
        _assert_no_overlap(placed)
        _assert_centered(root, placed)


def test_layout_is_idempotent() -> None:
    """Re-laying the same tree (a zoom re-render) gives identical positions."""
    root = LayoutInput(
        10,
        [LayoutInput(6, key="a"), LayoutInput(8, key="b")],
        key="r",
    )
    first = {p.key: p.x for p in layout(root)}
    second = {p.key: p.x for p in layout(root)}
    assert first == second


def test_bounds_reports_edges() -> None:
    root = LayoutInput(10, [LayoutInput(6, key="a"), LayoutInput(8, key="b")], key="r")
    placed = layout(root)
    min_left, max_right, max_depth = bounds(placed)
    assert min_left < max_right
    assert max_depth == 1
