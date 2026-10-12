"""A decorative fork may recover once, but never spend the foreground's route or budget."""

import asyncio
from typing import Any

import pytest

from local_operator.providers.failover import ProviderError, ProviderErrorKind
from local_operator.supplements import generator
from tests.unit.supplements.test_generator import (
    DATASETS,
    HONEST_BLOCK,
    _Usage,
    model_spec,
)


def _models():
    session = model_spec("openrouter", "openai/gpt-6.1-sol")
    return generator.resolve_design_model(
        configured="auto",
        session_spec=session,
        usable={"anthropic", "openrouter"},
        static_models=lambda provider: {"claude-sonnet-5-5": object()},
    )


def test_configured_hi_keeps_its_provider_and_outranks_ladder() -> None:
    tier = model_spec("openrouter", "openai/gpt-6.1-sol")
    chosen = generator.resolve_design_model(
        configured="auto",
        session_spec=model_spec(),
        tier_spec=tier,
        usable={"anthropic", "openrouter"},
        static_models=lambda provider: {"claude-sonnet-5-5": object()},
    )
    assert chosen.spec is tier and chosen.source == "tier"


def test_unknown_ladder_ids_are_never_guessed() -> None:
    session = model_spec("openrouter", "openai/gpt-6.1-sol")
    chosen = generator.resolve_design_model(
        configured="auto", session_spec=session, usable=None, static_models=lambda provider: {}
    )
    assert chosen.spec is session and chosen.source == "session"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,status",
    [("auth", 401), ("quota", 429), ("transient", 503), ("timeout", 504), ("request", 404)],
)
async def test_typed_failure_uses_one_distinct_isolated_alternate(
    kind: ProviderErrorKind, status: int
) -> None:
    requests = []

    async def complete(request: Any):
        requests.append(request)
        if len(requests) == 1:
            raise generator.CompletionFailure(
                ProviderError(
                    status,
                    "model unavailable" if status == 404 else "provider unavailable",
                    kind=kind,
                ),
                _Usage(10),
            )
        return HONEST_BLOCK, _Usage(10)

    outcome = await generator.generate(
        complete=complete,
        model=_models(),
        datasets=DATASETS,
        user_text="q",
        answer_text="a",
        max_turns=2,
        max_output_tokens=6000,
        max_cost_usd=1,
        price=lambda usage: 0.0,
    )
    assert len(requests) == 2 and outcome.components and not outcome.error
    assert requests[0].model.provider == "anthropic"
    assert requests[1].model.provider == "openrouter"
    assert all(request.isolated and not request.tools for request in requests)
    assert outcome.turns == 2 and outcome.model == "openrouter/openai/gpt-6.1-sol"
    assert len(outcome.attempts) == 2 and "status=" in " ".join(outcome.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", ["session", "anthropic/claude-sonnet-5-5"])
async def test_explicit_pin_never_substitutes(configured: str) -> None:
    chosen = generator.resolve_design_model(
        configured=configured,
        session_spec=model_spec(),
        usable=set(),
        static_models=lambda provider: {},
    )
    calls = []

    async def complete(request: Any):
        calls.append(request)
        raise ProviderError(429, "quota", kind="quota")

    outcome = await generator.generate(
        complete=complete,
        model=chosen,
        datasets=(),
        user_text="q",
        answer_text="a",
        max_turns=3,
        max_output_tokens=6000,
        max_cost_usd=1,
        price=lambda usage: 0.0,
    )
    assert len(calls) == 1 and outcome.error


@pytest.mark.asyncio
async def test_partial_usage_exhausts_cap_before_fallback() -> None:
    calls = []

    async def complete(request: Any):
        calls.append(request)
        raise generator.CompletionFailure(ProviderError(503, "unavailable"), _Usage(50))

    outcome = await generator.generate(
        complete=complete,
        model=_models(),
        datasets=(),
        user_text="q",
        answer_text="a",
        max_turns=3,
        max_output_tokens=6000,
        max_cost_usd=0.01,
        price=lambda usage: 0.02,
    )
    assert len(calls) == 1 and outcome.error == "bound:cost"
    assert outcome.tokens_in == 50 and outcome.cost_usd == 0.02


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["request", "unknown", "second_failure", "turn_cap", "deadline"])
async def test_recovery_never_becomes_a_retry_walk(case: str) -> None:
    requests = []

    async def complete(request: Any):
        requests.append(request)
        if case == "unknown":
            raise RuntimeError("not a provider failure")
        raise generator.CompletionFailure(
            ProviderError(
                400 if case == "request" else 429,
                "invalid argument" if case == "request" else "quota",
            ),
            _Usage(10),
        )

    outcome = await generator.generate(
        complete=complete,
        model=_models(),
        datasets=(),
        user_text="q",
        answer_text="a",
        max_turns=1 if case == "turn_cap" else 4,
        max_output_tokens=6000,
        max_cost_usd=1,
        price=lambda usage: 0.0,
        timeout_s=0 if case == "deadline" else None,
    )
    assert len(requests) == (2 if case == "second_failure" else 1)
    assert outcome.error and not outcome.components


def test_provider_diagnostic_never_carries_locations_or_credential_shapes() -> None:
    secret = "AKIA" + "IOSFODNN7EXAMPLE"
    text = generator.failure_detail(
        ProviderError(429, f"{secret} https://private.test/token /Users/private/file ~/hidden")
    )
    assert secret not in text and "private.test" not in text and "/Users" not in text
    assert "status=429 kind=quota" in text and len(text) < 400


@pytest.mark.asyncio
async def test_cancel_is_not_a_fallback_signal() -> None:
    async def complete(request: Any):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await generator.generate(
            complete=complete,
            model=_models(),
            datasets=(),
            user_text="q",
            answer_text="a",
            max_turns=2,
            max_output_tokens=6000,
            max_cost_usd=1,
            price=lambda usage: 0.0,
        )
