"""Deterministic path-health mining (``compaction.mine``): tiers, bounds, drift pin.

Two things must never drift here and so are pinned rather than asserted in prose:
the locator grammar must equal ``memory_capture._LOCATOR`` (they are independent
copies on purpose — see the module docstring for why the leaf cannot import the
heavier module), and the carry must be bounded by its tier caps regardless of how
much material is mined into it.
"""

from __future__ import annotations

import re

from local_operator.compaction.mine import (
    ASK_ITEM_CHARS,
    CARRY_ASK_ITEMS,
    CARRY_CHAR_BUDGET,
    CARRY_DECISION_ITEMS,
    CARRY_IDENTIFIER_ITEMS,
    LOCATOR_PATTERN,
    Carry,
    carry_digest_lines,
    carry_is_empty,
    merge_carry,
    mine_carry,
    parse_carry,
    render_carry,
)


def test_locator_grammar_is_pinned_to_memory_capture():
    """The two locator grammars are the SAME string, by test not by comment.

    ``mine`` cannot import ``memory_capture._LOCATOR`` (that would drag
    ``sqlite3`` and the redaction closure onto the compaction request path), so
    the grammar is copied. A silent divergence would capture different facts on
    the two paths and nothing would fail — this is the pin.
    """
    from local_operator import memory_capture

    assert LOCATOR_PATTERN == memory_capture._LOCATOR.pattern
    assert re.compile(LOCATOR_PATTERN).pattern == memory_capture._LOCATOR.pattern


def test_mine_attributes_asks_and_decisions_by_role_header():
    block = (
        "[User]\nplease keep the manager posting updates in thread 7\n"
        "[Assistant]\ndecided to keep the existing helper; do not add a new one\n"
        "[Tool result: bash]\n/Users/damian/local-operator/x.py and #1234\n"
    )
    carry = mine_carry(block)
    assert carry.asks == ["please keep the manager posting updates in thread 7"]
    assert carry.decisions == ["decided to keep the existing helper; do not add a new one"]
    # Identifiers are mined everywhere, not only from the speaker's own lines.
    assert "/Users/damian/local-operator/x.py" in carry.identifiers
    assert "#1234" in carry.identifiers


def test_mine_finds_no_ask_in_an_assistant_line():
    """An assistant line that merely mentions a user is not an operator ask."""
    carry = mine_carry("[Assistant]\nthe user asked about the deploy earlier\n")
    assert carry.asks == []
    assert carry.decisions == []


def test_long_line_is_scanned_bounded_and_publishes_no_fragment():
    """A huge line cannot make the scan quadratic nor publish half a token."""
    blob = "/Users/damian/" + ("a" * 40) + "/deep.py " + ("x" * 100_000)
    carry = mine_carry(f"[Tool result: bash]\n{blob}\n")
    assert "/Users/damian/" + ("a" * 40) + "/deep.py" in carry.identifiers
    assert all(len(item) <= 200 for item in carry.identifiers)
    assert all(item != "//" for item in carry.identifiers)


def test_render_and_parse_round_trip():
    carry = Carry(identifiers=["#1", "/a/b.py"], asks=["do X"], decisions=["decided Y"])
    rendered = render_carry(carry)
    assert parse_carry(rendered) == carry
    assert parse_carry(render_carry(parse_carry(rendered))) == carry


def test_empty_carry_renders_empty_and_is_empty():
    assert render_carry(Carry()) == ""
    assert carry_is_empty(Carry())
    assert parse_carry("") == Carry()


def test_identifiers_are_never_evicted_inside_the_cap():
    """Tier ranking: the identifier tier survives even when asks+decisions flood.

    This is the fix for the cliff: age-ordered eviction only DELAYS the loss
    (measured), so the eviction order must be tier-first. Identifiers are the
    retention anchors a resumed session is checked against, so they go LAST.
    """
    prior = Carry(identifiers=["#100", "/keep/me.py"], asks=[], decisions=[])
    fresh = Carry(
        asks=[f"ask {i} " + "a" * 200 for i in range(CARRY_ASK_ITEMS * 2)],
        decisions=[f"decision {i} " + "d" * 200 for i in range(CARRY_DECISION_ITEMS * 2)],
    )
    merged = merge_carry(prior, fresh)
    assert "#100" in merged.identifiers
    assert "/keep/me.py" in merged.identifiers
    assert len(merged.identifiers) <= CARRY_IDENTIFIER_ITEMS
    assert len(merged.asks) <= CARRY_ASK_ITEMS
    assert len(merged.decisions) <= CARRY_DECISION_ITEMS


def test_carry_is_always_bounded_by_its_char_budget():
    """However much is mined, the rendered carry cannot exceed the budget."""
    carry = Carry()
    for i in range(400):
        carry = merge_carry(
            carry,
            Carry(
                asks=[f"ask {i} " + "a" * ASK_ITEM_CHARS],
                decisions=[f"decision {i} " + "d" * 300],
                identifiers=[f"/path/{i}/x.py"],
            ),
        )
    rendered = render_carry(carry)
    assert len(rendered) <= CARRY_CHAR_BUDGET


def test_merge_is_additive_and_order_preserving():
    prior = Carry(asks=["first"], decisions=["d1"], identifiers=["#1"])
    fresh = Carry(asks=["second"], decisions=["d2"], identifiers=["#2"])
    merged = merge_carry(prior, fresh)
    assert merged.asks == ["first", "second"]
    assert merged.decisions == ["d1", "d2"]
    assert merged.identifiers == ["#1", "#2"]


def test_digest_lines_are_bounded_and_tier_labelled():
    carry = Carry(asks=["ask one"], decisions=["dec one"], identifiers=["#9"])
    lines = carry_digest_lines(carry, max_chars=1000)
    assert any(line.startswith("- Asks:") for line in lines)
    assert any(line.startswith("- Decisions:") for line in lines)
    assert any(line.startswith("- Identifiers:") for line in lines)
    assert len("\n".join(lines)) <= 1000
