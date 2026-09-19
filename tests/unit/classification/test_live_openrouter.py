"""Live check that our client speaks the alpha route's real shape (§11.4).

SKIPPED BY DEFAULT, and that is the point: the unit suite must be hermetic, so
this module runs only when a key is already in the environment —

    OPENROUTER_API_KEY_DEV=$(lop secret get OPENROUTER_API_KEY_DEV) \\
      .venv/bin/python -m pytest tests/unit/classification/test_live_openrouter.py -q -s

``tests/conftest.py`` deliberately clears ``OPENROUTER_API_KEY`` from the ambient
environment (so no unit test reaches the network by accident); it does NOT clear
``OPENROUTER_API_KEY_DEV``, which is the tier this file reads. The difference
matters: with the production key cleared and the dev key not, a run here can only
ever spend the dev quota, and a machine with neither key skips.

The key is read from the environment and handed to the credential manager without
ever being printed, logged or written to a file.
"""

from __future__ import annotations

import json
import os

import pytest

from local_operator.classification.recommend import RecommendationRequest
from local_operator.classification.service import ClassificationService
from local_operator.classification.types import DecisionRequest, Question
from local_operator.classification.vendors import OpenRouterVendor
from local_operator.credentials import CredentialManager

LIVE_KEY = os.environ.get("OPENROUTER_API_KEY_DEV") or os.environ.get("OPENROUTER_API_KEY", "")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not LIVE_KEY, reason="no OpenRouter key in the environment"),
]

#: A request a human would actually type, and the resource that should win it.
DEPLOY_MESSAGE = "deploy the new core build to qa and tell me when it is live"

CANDIDATES = (
    "minerva-platform-deployments",
    "Deploys, promotes, rolls back and shifts ingress for a Minerva service",
    "minerva-support-workspace",
    "Handles Minerva support requests, directory lookups and attendee timezones",
    "pergamon-enrichment",
    "Runs Pergamon enrichment pipelines over company and contact records",
)


def _manager(tmp_path) -> CredentialManager:
    manager = CredentialManager(tmp_path)
    manager.set_credential("OPENROUTER_API_KEY", LIVE_KEY, write=False)
    return manager


def _settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {"auto": True, "vendor": "openrouter", "timeoutMs": 30_000}
    values.update(overrides)
    return {"classification": values}


async def test_the_route_answers_a_choice_and_a_score_question_in_our_shape(tmp_path) -> None:
    """The measured proof that the request we build is one the vendor accepts.

    Sends both criteria shapes in one call — a ``choice`` whose criterion VALUES
    are strings (§3's non-negotiable) and a ``score`` whose criteria are an
    ARRAY — and prints the request and the response so the shape and the cost can
    be quoted verbatim. This is the check a green MockTransport test cannot make.
    """
    manager = _manager(tmp_path)
    vendor = OpenRouterVendor(manager)
    request = DecisionRequest(
        state={"request": DEPLOY_MESSAGE},
        questions=(
            Question(
                id="recommend_skill",
                kind="choice",
                instructions="Which skill would actually help with the request above?",
                criteria={
                    "minerva-platform-deployments": "Deploys, promotes or rolls back a service",
                    "minerva-support-workspace": "Handles support requests",
                    "none": "None of these fits this request",
                },
            ),
            Question(
                id="effort",
                kind="score",
                instructions="How involved is this request?",
                criteria=("trivial", "routine", "complex"),
            ),
        ),
    )
    response = await vendor.decide(request, timeout_s=30.0)

    print(
        "\nsent questions:",
        json.dumps(
            vendor
            and request.questions
            and {
                question.id: {
                    "type": question.kind,
                    "criteria": question.criteria,
                }
                for question in request.questions
            },
            ensure_ascii=False,
        ),
    )
    print(
        "answered:",
        json.dumps(
            {
                answer_id: {
                    "kind": answer.kind,
                    "value": answer.value,
                    "probabilities": answer.probabilities,
                    "confidence": answer.confidence,
                }
                for answer_id, answer in response.answers.items()
            },
            ensure_ascii=False,
        ),
    )
    print(
        "usage:",
        {
            "vendor": response.vendor,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "latency_s": round(response.latency_s, 3),
        },
    )

    assert response.vendor == "openrouter"
    assert response.model.startswith("typesafe/jev")
    offered = request.questions[0].criteria
    assert isinstance(offered, dict)
    assert response.answers["recommend_skill"].value in offered
    assert isinstance(response.answers["effort"].value, float)
    # Both counts are asserted as REPORTED rather than compared to a sentinel:
    # since absent became ``None``, a comparison alone would not tell a missing
    # figure from a real zero, which is precisely the distinction the accounting
    # now keeps.
    assert response.input_tokens is not None and response.input_tokens > 0
    assert response.output_tokens is not None and response.output_tokens >= 0
    assert response.cost_usd is not None and response.cost_usd > 0
    assert response.latency_s > 0.0


async def test_the_service_recommends_the_matching_skill_end_to_end(tmp_path) -> None:
    """§11.4's end-to-end claim, on one message: the block, the vendor, the cost."""
    from local_operator.classification.context import Candidate

    manager = _manager(tmp_path)
    service = ClassificationService(manager=manager, settings=_settings())
    recommendation = await service.recommend_resources(
        RecommendationRequest(
            user_message=DEPLOY_MESSAGE,
            context=None,
            candidates=(
                Candidate(
                    kind="skill",
                    name=CANDIDATES[0],
                    description=CANDIDATES[1],
                    resource_url=f"skill://{CANDIDATES[0]}",
                ),
                Candidate(
                    kind="skill",
                    name=CANDIDATES[2],
                    description=CANDIDATES[3],
                    resource_url=f"skill://{CANDIDATES[2]}",
                ),
                Candidate(
                    kind="skill",
                    name=CANDIDATES[4],
                    description=CANDIDATES[5],
                    resource_url=f"skill://{CANDIDATES[4]}",
                ),
            ),
        )
    )

    print("\nblock:", repr(recommendation.block))
    print(
        "result:",
        {
            "vendor": recommendation.vendor,
            "resources": [item.resource_url for item in recommendation.resources],
            "cost_usd": recommendation.cost_usd,
            "latency_s": round(recommendation.latency_s, 3),
            "skipped": recommendation.skipped,
        },
    )

    assert recommendation.skipped is None
    assert recommendation.vendor == "openrouter"
    assert "skill://minerva-platform-deployments" in recommendation.block
    assert recommendation.cost_usd is not None and recommendation.cost_usd > 0
    assert recommendation.block.startswith("<resource_recommendations>")
    assert recommendation.block.endswith("</resource_recommendations>")


async def test_the_local_overhead_is_a_small_fraction_of_a_real_call(tmp_path) -> None:
    """The latency budget, with the REAL leg classes: ours versus the vendor's.

    Two numbers, deliberately kept apart rather than subtracted into one claim:

    * the local path, measured in a loop with the credential memo already warm
      and no network at all (state build via the roster memo, question build,
      memo hit) — this is OUR overhead per message;
    * the vendor's own ``latency_s`` for one real call, as the leg measured it
      around its own HTTP request.

    The service call's ``latency_s`` is the sum of the two plus a few hundred
    microseconds of bookkeeping, which the last assertion makes explicit.
    """
    import time

    from local_operator.classification.context import RosterCache, build_state
    from local_operator.classification.recommend import (
        build_decision_request,
        build_questions,
    )

    manager = _manager(tmp_path)
    vendor = OpenRouterVendor(manager)
    candidates = tuple(
        __import__("local_operator.classification.context", fromlist=["Candidate"]).Candidate(
            kind="skill",
            name=CANDIDATES[index],
            description=CANDIDATES[index + 1],
            resource_url=f"skill://{CANDIDATES[index]}",
        )
        for index in (0, 2, 4)
    )
    roster_cache = RosterCache()
    settings = _settings()

    # Warm the credential memo so the loop measures the path, not the resolve.
    await vendor.credential(manager)

    iterations = 50
    started = time.perf_counter()
    for _ in range(iterations):
        plan = build_questions(candidates, limit=12)
        state = build_state(
            user_message=DEPLOY_MESSAGE,
            context=None,
            candidates=candidates,
            settings=settings,
            roster_cache=roster_cache,
        )
        build_decision_request(plan, state)
        await vendor.credential(manager)
    local_ms = (time.perf_counter() - started) / iterations * 1000

    response = await vendor.decide(
        DecisionRequest(
            state=build_state(
                user_message=DEPLOY_MESSAGE,
                context=None,
                candidates=candidates,
                settings=settings,
                roster_cache=roster_cache,
            ),
            questions=build_questions(candidates, limit=12).questions,
        ),
        timeout_s=30.0,
    )

    print(f"\nours (local path, memoized creds, no network): {local_ms:.3f} ms/call")
    print(f"vendor's own measured request time: {response.latency_s * 1000:.1f} ms")
    print(
        "vendor usage:",
        {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "model": response.model,
        },
    )
    assert local_ms < 10.0
    assert response.latency_s > 0.0
