"""Unknown spend never buys another paid invocation; explicit pre-wire refusal is free."""

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.providers.failover import ProviderError
from local_operator.session.runtime.supplements import SupplementRunner
from local_operator.supplements import generator
from tests.unit.supplements.test_generator import _Usage, model_spec
from tests.unit.supplements.test_generator_reliability import _models


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "unknown_price", "pricing_error"])
async def test_unknown_paid_spend_stops_before_alternate(case: str) -> None:
    calls = []

    async def complete(request: Any):
        calls.append(request)
        raise generator.CompletionFailure(
            ProviderError(503, "unavailable"), None if case == "missing" else _Usage(50000)
        )

    def price(usage: Any):
        if case == "pricing_error":
            raise ValueError("pricing unavailable")
        return None

    outcome = await generator.generate(
        complete=complete,
        model=_models(),
        datasets=(),
        user_text="q",
        answer_text="a",
        max_turns=2,
        max_output_tokens=1000,
        max_cost_usd=0.01,
        model_price=lambda usage, model: price(usage),
    )
    assert len(calls) == 1 and outcome.error == "bound:cost" and not outcome.cost_known
    assert outcome.tokens_in == (0 if case == "missing" else 50000)


@pytest.mark.asyncio
async def test_certified_pre_wire_refusal_does_not_hide_unknown_paid_usage() -> None:
    calls = []

    async def complete(request: Any):
        calls.append(request)
        if len(calls) == 1:
            raise generator.CompletionFailure(
                ProviderError(None, "credential blocked", kind="quota", request_sent=False), None
            )
        return "NONE", _Usage(10)

    outcome = await generator.generate(
        complete=complete,
        model=_models(),
        datasets=(),
        user_text="q",
        answer_text="a",
        max_turns=2,
        max_output_tokens=1000,
        max_cost_usd=0.01,
        price=lambda usage: 0,
    )
    assert len(calls) == 2 and not outcome.error and outcome.cost_known


def test_pricer_keeps_provider_receipt_and_explicit_free_vs_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.model import configure
    from local_operator.model.registry import unknown_model_info

    monkeypatch.setattr(configure, "resolve_model_info", lambda *_args: unknown_model_info)
    runner = object.__new__(SupplementRunner)
    paid = generator.DesignModel(
        model_spec("openrouter", "unknown-paid"), "config", "openrouter/unknown-paid"
    )
    assert runner._price(SimpleNamespace(usd_cost=0.02), paid) == 0.02
    assert runner._price(SimpleNamespace(usd_cost=0), paid) == 0
    assert runner._price(_Usage(50000), paid) is None
    assert runner._price(None, paid) is None
    free = generator.DesignModel(model_spec("test", "fixture"), "session", "test/fixture")
    assert runner._price(_Usage(10), free) == 0
