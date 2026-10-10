"""The rung DATA table: order, labels and specs stay consistent by test.

Rungs are data now (design D10): one order constant and one spec table drive
the resolver, and the checklist that adds a rung edits them together. These
tests pin the join so the three authorities — the order, the label mapping and
the spec table — cannot drift; the per-rung VALUES are pinned per rung
(cancel support is declaration-only in v1, so a wrong value here misleads
without failing a call).
"""

from __future__ import annotations

from local_operator.artifacts.rung import CancelSupport
from local_operator.imagegen import ImageRoute, cascade


def test_the_v1_order_is_frozen_and_append_only() -> None:
    # The three v1 routes keep their exact positions; wave-2 rungs append.
    assert cascade.IMAGE_RUNG_ORDER[:3] == (
        ImageRoute.RADIENT,
        ImageRoute.FAL,
        ImageRoute.OPENAI,
    )
    assert ImageRoute.NONE not in cascade.IMAGE_RUNG_ORDER


def test_every_ordered_route_has_a_spec_and_a_label() -> None:
    assert set(cascade.RUNG_SPECS) == set(cascade.IMAGE_RUNG_ORDER)
    assert set(cascade.RUNG_LABELS) >= set(cascade.IMAGE_RUNG_ORDER)
    for route in cascade.IMAGE_RUNG_ORDER:
        spec = cascade.RUNG_SPECS[route]
        assert spec.route == route, "the spec's wire spelling must BE the route"
        assert spec.label == cascade.RUNG_LABELS[route], "labels must not drift from specs"
        assert spec.kinds == frozenset({"image"})
        assert spec.cancel_support in CancelSupport


def test_the_v1_spec_facts_are_pinned() -> None:
    radient = cascade.RUNG_SPECS[ImageRoute.RADIENT]
    assert radient.capabilities == frozenset({"t2i", "i2i"})
    assert radient.cancel_support == CancelSupport.SIGNAL
    assert radient.cost == "reported"

    fal = cascade.RUNG_SPECS[ImageRoute.FAL]
    assert fal.capabilities == frozenset({"t2i", "i2i"})
    # Per best available evidence: FAL's cancel URL answers
    # CANCELLATION_REQUESTED; mid-run honour is UNVERIFIED. Declaration-only
    # in v1 — no branch reads this value.
    assert fal.cancel_support == CancelSupport.SIGNAL
    assert fal.cost == "rate_table"

    openai = cascade.RUNG_SPECS[ImageRoute.OPENAI]
    assert openai.capabilities == frozenset({"t2i"}), "img2img is skipped on this rung"
    assert openai.cancel_support == CancelSupport.NONE
    assert openai.cost == "rate_table"
