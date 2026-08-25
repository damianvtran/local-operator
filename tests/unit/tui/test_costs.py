"""Per-session and aggregate cost accessors for the TUI's money surfaces.

The owner's report was that a delegated turn's spend is invisible: it goes on
the CHILD's model, which may differ from the parent's, and the band showed only
what the parent itself burned. These cover the accessors the band, the subagent
panel and the full-page view all read.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from local_operator.harness.types import Usage
from local_operator.model.registry import ModelInfo
from local_operator.tui.costs import job_cost, turn_cost

#: $10 in / $100 out per MTok, so a million tokens is a round number.
_PRICED = ModelInfo(id="m", name="m", description="", input_price=10.0, output_price=100.0)
#: A real model the registry simply has no price for.
_UNPRICED = ModelInfo(id="m", name="m", description="")


def _resolving(**by_model_id: ModelInfo):
    """Patch model resolution to answer from ``by_model_id``.

    Keyed on the model id alone, which is what the ``provider/model_id`` label
    splits into; an unknown id resolves unpriced, so a test can assert the
    "no published price" path without inventing a provider. Both resolvers
    are patched: ``turn_cost`` prices through the paint-safe one
    (:func:`~local_operator.model.configure.resolve_model_info_paint`), and a
    test that patched only the full resolver would exercise the real disk
    path instead of its own table — the assertion would still pass or fail,
    but on the wrong evidence.
    """
    return patch.multiple(
        "local_operator.model.configure",
        resolve_model_info=lambda provider, model_id: by_model_id.get(model_id, _UNPRICED),
        resolve_model_info_paint=lambda provider, model_id: (
            by_model_id.get(model_id, _UNPRICED),
            True,
        ),
    )


def _job(job_id: str, *, usage: Usage | None, model_label: str | None = None) -> SimpleNamespace:
    """An ``AsyncJob``-shaped stand-in; the accessors are duck-typed."""
    return SimpleNamespace(id=job_id, usage=usage, model_label=model_label)


# ---------------------------------------------------------------------------
# turn_cost
# ---------------------------------------------------------------------------


def test_turn_cost_prices_a_turn_on_its_own_model() -> None:
    with _resolving(opus=_PRICED):
        cost = turn_cost("anthropic/opus", Usage(input_tokens=1_000_000, output_tokens=100_000))
    assert cost == pytest.approx(10.0 + 10.0)


def test_turn_cost_is_none_for_an_unpriced_model() -> None:
    """``None``, never ``0.0``: a confident $0.0000 reads as "that was free"."""
    with _resolving():
        assert turn_cost("anthropic/mystery", Usage(input_tokens=1_000_000)) is None


def test_turn_cost_is_none_without_usage_or_a_label() -> None:
    with _resolving(opus=_PRICED):
        assert turn_cost("anthropic/opus", None) is None
        assert turn_cost("", Usage(input_tokens=10)) is None


def test_turn_cost_survives_a_broken_resolver() -> None:
    """A pricing failure degrades to "unknown", never to a broken frame."""
    with patch(
        "local_operator.model.configure.resolve_model_info_paint",
        side_effect=RuntimeError("boom"),
    ):
        assert turn_cost("anthropic/opus", Usage(input_tokens=10)) is None


def test_turn_cost_prefers_a_provider_reported_dollar() -> None:
    """OpenRouter's precomputed bill is authoritative and must win verbatim over
    the model's table price — the routed provider's actual rate is not the table."""
    with _resolving(opus=_PRICED):
        cost = turn_cost(
            "anthropic/opus", Usage(input_tokens=1_000_000, output_tokens=100_000, usd_cost=0.0075)
        )
    assert cost == pytest.approx(0.0075)


def test_turn_cost_reported_dollar_wins_even_without_a_table_price() -> None:
    """The provider's bill is the fact the table is an estimate of, so it must
    answer even when the model has no published price row — a turn the provider
    billed must never read as unpriceable."""
    usage = Usage(input_tokens=10, output_tokens=16, usd_cost=0.0000075)
    assert turn_cost("openrouter/deepseek/deepseek-v4-flash-0731", usage) == pytest.approx(
        0.0000075
    )


def test_turn_cost_never_renders_a_reported_credit() -> None:
    """A negative reported amount is malformed provider data, not a refund: it
    must fall back to the table estimate rather than paint an upside-down dollar
    figure on the band."""
    with _resolving(opus=_PRICED):
        usage = Usage(input_tokens=1_000_000, output_tokens=0, usd_cost=-5.0)
        assert turn_cost("anthropic/opus", usage) == pytest.approx(10.0)


def test_turn_cost_reported_malformed_falls_back_to_estimate() -> None:
    """A non-numeric reported amount from a rehydrated mapping must not degrade
    the whole turn to unpriceable when a table price exists."""
    with _resolving(opus=_PRICED):
        usage = {"input_tokens": 1_000_000, "output_tokens": 0, "usd_cost": "not-a-number"}
        assert turn_cost("anthropic/opus", usage) == pytest.approx(10.0)


def test_turn_cost_non_finite_reported_falls_back_to_estimate() -> None:
    """A non-finite reported amount (``inf``/``NaN``) is wire-reachable via
    ``json.loads`` and, left unfloored, renders an infinite dollar figure. It
    must fall back to the finite table estimate rather than paint infinity."""
    import math

    for bad in (float("inf"), float("-inf"), float("nan")):
        with _resolving(opus=_PRICED):
            usage = Usage(input_tokens=1_000_000, output_tokens=0, usd_cost=bad)
            result = turn_cost("anthropic/opus", usage)
            assert result == pytest.approx(10.0)
            assert result is not None and math.isfinite(result)


# ---------------------------------------------------------------------------
# job_cost — one child
# ---------------------------------------------------------------------------


def test_job_cost_uses_the_parents_model_when_the_child_recorded_none() -> None:
    """The common case: a child inherits the parent's spec, so it records nothing."""
    job = _job("a", usage=Usage(input_tokens=1_000_000))
    with _resolving(opus=_PRICED):
        assert job_cost(job, default_model_label="anthropic/opus") == pytest.approx(10.0)


def test_job_cost_prices_a_child_on_its_own_different_model() -> None:
    """A `model_spec` override means the child's tokens are NOT the parent's rate.

    The whole reason the label is recorded per job: pricing a cheap child at the
    parent's flagship rate overstates a fan-out by an order of magnitude, and
    pricing an expensive one at a cheap parent's rate hides real money.
    """
    cheap = ModelInfo(id="h", name="h", description="", input_price=1.0, output_price=5.0)
    job = _job("a", usage=Usage(input_tokens=1_000_000), model_label="anthropic/haiku")
    with _resolving(opus=_PRICED, haiku=cheap):
        assert job_cost(job, default_model_label="anthropic/opus") == pytest.approx(1.0)


def test_job_cost_is_none_when_the_child_has_reported_no_usage() -> None:
    """A bash job, or a child that has not finished a model call yet.

    ``None`` rather than ``0.0`` because "spent nothing" and "has not told us
    yet" are different facts, and only one of them is worth a number on screen.
    """
    with _resolving(opus=_PRICED):
        assert job_cost(_job("a", usage=None), default_model_label="anthropic/opus") is None


def test_job_cost_is_zero_for_a_priced_child_that_burned_nothing() -> None:
    """A real zero is a fact, and is distinct from the ``None`` above."""
    job = _job("a", usage=Usage(input_tokens=0, output_tokens=0))
    with _resolving(opus=_PRICED):
        assert job_cost(job, default_model_label="anthropic/opus") == 0.0


def test_job_cost_survives_a_job_whose_fields_raise() -> None:
    """The accessors are duck-typed, so the field READS are guarded too.

    The TUI runs against embedder hosts and replayed ledgers whose job objects
    are not ``AsyncJob``, so ``usage`` can be a property with real work behind
    it. An exception escaping here takes down a band repaint.
    """

    class Hostile:
        @property
        def usage(self) -> Usage:
            raise RuntimeError("this ledger row cannot be read")

    assert job_cost(Hostile(), default_model_label="anthropic/opus") is None
