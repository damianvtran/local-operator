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


def test_identifier_eviction_is_newest_kept_like_the_other_tiers():
    """MAJOR 3: identifiers must not silently keep the OLDEST when full.

    An earlier revision frequency-ranked identifiers and sliced ``[:cap]``,
    dropping the NEWEST references while asks/decisions kept newest — the
    asymmetry that lost the reference stated on the current task. All tiers now
    keep newest uniformly.
    """
    cap = CARRY_IDENTIFIER_ITEMS
    carry = Carry()
    for i in range(cap + 30):
        carry = merge_carry(carry, Carry(identifiers=[f"/path/step{i:03d}/x.py"]))
    assert len(carry.identifiers) == cap
    assert "/path/step000/x.py" not in carry.identifiers  # oldest evicted
    assert f"/path/step{cap + 29:03d}/x.py" in carry.identifiers  # newest kept


def test_mined_identifiers_keep_first_seen_order():
    """The miner must emit identifiers chronologically, not by frequency, or the
    newest-kept merge slice keeps the oldest (MAJOR 3)."""
    text = "[Tool result: bash]\n#999 #999 #999 /old.py\n" "[Tool result: bash]\n/new.py\n"
    carry = mine_carry(text)
    assert carry.identifiers.index("/old.py") < carry.identifiers.index("/new.py")


def test_carry_is_never_re_mined_as_history():
    """MAJOR 2: a replayed carry must not be re-mined into asks (self-echo).

    The carry replays as leading text in the next pass's ``to_summarize``; if
    the miner re-reads it, every ``- `` row under the replayed block becomes an
    ask and the digest nests and evicts the genuine asks.
    """
    carried = (
        "[User]\n## Operator asks\n- old ask one\n- old ask two\n"
        "## Identifiers\n- /keep/me.py\n"
        "[Assistant]\nreminder note\n"
    )
    carry = mine_carry(carried)
    assert carry.asks == []
    assert carry.identifiers == []
    # And a NON-carry history with the same words still mines normally.
    plain = "[User]\nplease do the thing in thread 7\n"
    assert mine_carry(plain).asks == ["please do the thing in thread 7"]


def test_forced_cut_never_publishes_a_fragment():
    """MAJOR 4: a token bisected by ``_MAX_LINE_SCAN`` is refused, not truncated."""
    from local_operator.compaction.mine import _MAX_LINE_SCAN

    uri = "https://example.com/" + "a" * (_MAX_LINE_SCAN + 50)
    carry = mine_carry(f"[Tool result: bash]\nsee {uri}\n")
    assert carry.identifiers == []  # not a 200-char prefix of the URI


def test_carry_render_includes_headings_in_the_budget():
    """MINOR 6: the rendered digest fits ``CARRY_CHAR_BUDGET`` with headings."""
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
    assert len(render_carry(carry)) <= CARRY_CHAR_BUDGET


def test_tool_error_line_is_not_an_assistant_decision():
    """NIT 9: a tool ERROR line is not a stated assistant decision."""
    carry = mine_carry("[Tool ERROR: bash]\nnever delete the production bucket\n")
    assert carry.decisions == []


def test_digest_lines_are_bounded_and_tier_labelled():
    carry = Carry(asks=["ask one"], decisions=["dec one"], identifiers=["#9"])
    lines = carry_digest_lines(carry, max_chars=1000)
    assert any(line.startswith("- Asks:") for line in lines)
    assert any(line.startswith("- Decisions:") for line in lines)
    assert any(line.startswith("- Identifiers:") for line in lines)
    assert len("\n".join(lines)) <= 1000
