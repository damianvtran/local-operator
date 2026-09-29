from __future__ import annotations

import pytest

from local_operator.hub_sync.segment import (
    align,
    norm,
    render,
    segment,
    split_sentences,
)


def test_norm_is_crlf_blank_and_trailing_space_insensitive() -> None:
    assert norm("a  \r\n\r\n\r\n\r\nb\r\n") == "a\n\nb"
    assert norm(None) == ""


def test_regions_atoms_and_fences() -> None:
    regions = segment("pre.\n\n## A\n- one\n- two\n\n```py\nx = 1\n\ny = 2\n```\n| a | b |")
    assert [r.key for r in regions] == ["", "a#0"]
    kinds = [a.kind for a in regions[1].atoms]
    assert kinds == ["item", "item", "fence", "row"]
    # A blank line inside a fence does not split it.
    assert "y = 2" in regions[1].atoms[2].text


def test_heading_inside_a_fence_is_not_a_region() -> None:
    assert len(segment("## A\n```\n## not a heading\n```")) == 1


def test_duplicate_headings_get_distinct_stable_keys() -> None:
    assert [r.key for r in segment("## A\nx.\n## A\ny.")] == ["a#0", "a#1"]


def test_a_demotion_keeps_the_region_key() -> None:
    assert segment("## A\nx.")[0].key == segment("### A\nx.")[0].key


def test_sentence_split_ignores_code_spans_and_needs_a_capital() -> None:
    assert [s for _, s in split_sentences("Use `a. B` here. Then stop.")] == [
        "Use `a. B` here.",
        "Then stop.",
    ]
    assert len(split_sentences("see e.g. this thing")) == 1


@pytest.mark.parametrize(
    "text",
    [
        "## A\nx.\n## B\ny.",
        "pre text. More here.\n\n## A\n- a\n- b\n\n```py\nx\n```\n| a | b |\n\n## A\nz.",
        "just one paragraph. Two sentences.",
    ],
)
def test_render_is_the_inverse_of_segment(text: str) -> None:
    assert render(segment(text)) == norm(text)


def test_align_matches_exact_then_similar_and_never_across_kinds() -> None:
    a = segment("## A\nBe brief.\nCite sources.")[0].atoms
    b = segment("## A\nCite sources.\nBe brief and cite sources.")[0].atoms
    pairs = align(a, b).pairs
    assert pairs[1] == (0, False)  # exact
    assert pairs[0] == (1, True)  # edited
    row = segment("| a | b |")[0].atoms
    assert align(row, segment("a b")[0].atoms).pairs == {}
