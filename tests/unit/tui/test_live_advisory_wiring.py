"""The advisory reaches the CARD, not just the tool result (review M5).

Two producers write a bounded-work advisory into an update's ``details``: the
memory guard (``memory_advisory``) and the soft query budget
(``query_budget_advisory``). Only the first had a reader, so during the 10-60 s
window the human saw nothing new and the PR's "advisory on both channels" claim
was true of the result and false of the screen.

These cells pin the WIRING — :func:`_partial_advisory` is what
``on_tool_updated`` hands to ``card.set_live_advisory`` — and then the rendering
half is pinned where the card lives (``test_tool_card.py``), so a regression in
either half fails a named test rather than a screenshot nobody looks at.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.tui.app import _partial_advisory

MEMORY = "memory 2.7/3.2 GB — approaching the command budget"
QUERY = "QUERY BUDGET: this shell query (`find`) has run 12s."


def _update(**details: object) -> SimpleNamespace:
    """A partial-result shape carrying ``details`` — what the stream sends."""
    return SimpleNamespace(details=details)


def test_the_query_budget_advisory_reaches_the_card() -> None:
    """The exact gap: a query-shaped command sets ONLY this key, and before the
    reader knew about it the card painted nothing."""
    assert _partial_advisory(_update(query_budget_advisory=QUERY)) == QUERY


def test_the_memory_advisory_still_wins_when_both_are_set() -> None:
    """A fixed rank, because the card has ONE state line: the device being short
    of RAM is the more urgent fact, and an unordered reader would flicker."""
    assert _partial_advisory(_update(memory_advisory=MEMORY, query_budget_advisory=QUERY)) == MEMORY
    # Either one alone still lands.
    assert _partial_advisory(_update(memory_advisory=MEMORY)) == MEMORY


@pytest.mark.parametrize(
    "details",
    [
        {},
        {"query_budget_advisory": None},
        {"query_budget_advisory": ""},
        {"query_budget_advisory": 17},
    ],
)
def test_a_missing_or_malformed_advisory_reads_as_none(details: dict[str, object]) -> None:
    """A card must never fail to render because a producer sent a shape it did
    not expect — the contract the memory advisory already had."""
    assert _partial_advisory(_update(**details)) is None


def test_absent_details_reads_as_none() -> None:
    assert _partial_advisory(SimpleNamespace()) is None
    assert _partial_advisory(SimpleNamespace(details=None)) is None
    assert _partial_advisory(SimpleNamespace(details=[])) is None
