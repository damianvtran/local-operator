"""The per-message latency budget, measured rather than claimed.

Why this file exists: this layer runs once per user message, on the critical
path, before the turn's first token. The design constraint is therefore "our own
overhead stays small", and the only honest way to hold that is a test that
exercises the local path and prints the number.

WHAT IS MEASURED, AND WHAT IS DELIBERATELY NOT
==============================================

The local path only: settings reads, the cache key, the roster memo, the state
build, the question build, the credential memo hit and the serialization. The
vendor's model time (~200 ms, measured) is not ours and is not in the budget.

What this file measures is OUR OWN overhead only — not how long a turn waits,
which is the caller's knob (§7's wiring waits ``values.classification.waitMs``,
default 50 ms, and delivers a late answer on a later turn).

The ceiling asserted here is **10 ms**, deliberately far above the measured
0.011-0.2 ms: this suite runs on a shared machine with a dozen concurrent
worktrees, and a budget test that flakes on a loaded host would be deleted
rather than fixed. 10 ms still catches a real regression — anything that
reintroduces per-message credential resolution, a per-message TLS handshake or
per-message roster serialization costs milliseconds, not microseconds — without
punishing the host's mood. The measured figure travels in the PR report and in
the service docstring.
"""

from __future__ import annotations

import time

import pytest

from local_operator.classification.recommend import RecommendationRequest
from tests.unit.classification.support import candidate, choice_response

pytestmark = pytest.mark.asyncio

#: The assertion ceiling (see the module docstring for why it is generous).
BUDGET_CEILING_S = 0.010

ITERATIONS = 200

ROSTER = [
    candidate(f"skill-{index}", description="A harness-owned description of a resource")
    for index in range(12)
] + [
    candidate(f"guide-{index}", kind="guide", description="A harness-owned description")
    for index in range(12)
]


def settings() -> dict[str, object]:
    return {"classification": {"auto": True}}


def request(message: str) -> RecommendationRequest:
    return RecommendationRequest(user_message=message, context=None, candidates=ROSTER)


async def test_the_local_path_is_far_under_the_budget(manager, install_legs) -> None:
    """A fresh message each time: cache MISS, so the full local path runs."""
    behaviours = install_legs(radient={"script": [choice_response("recommend_skill", "skill-0")]})
    from local_operator.classification.service import ClassificationService

    subject = ClassificationService(manager=manager, settings=settings())
    await subject.recommend_resources(request("warm up the memos"))  # credential + roster memos

    started = time.perf_counter()
    for index in range(ITERATIONS):
        await subject.recommend_resources(request(f"message number {index}"))
    per_call_s = (time.perf_counter() - started) / ITERATIONS

    print(
        f"\nlocal path (cache miss, credential memo warm, roster memo warm): "
        f"{per_call_s * 1000:.3f} ms/call over {ITERATIONS} calls; "
        f"vendor calls: {len(behaviours['radient'].calls)}"
    )
    assert per_call_s < BUDGET_CEILING_S


async def test_a_cache_hit_is_much_cheaper_than_the_budget(manager, install_legs) -> None:
    behaviours = install_legs(radient={"script": [choice_response("recommend_skill", "skill-0")]})
    from local_operator.classification.service import ClassificationService

    subject = ClassificationService(manager=manager, settings=settings())
    warm = request("deploy core to qa")
    await subject.recommend_resources(warm)  # fills the cache

    started = time.perf_counter()
    for _ in range(ITERATIONS):
        await subject.recommend_resources(warm)
    per_call_s = (time.perf_counter() - started) / ITERATIONS

    calls = len(behaviours["radient"].calls)
    print(f"\ncache hit: {per_call_s * 1000:.4f} ms/call; vendor calls: {calls}")
    # One vendor call in total: the other 200 went nowhere near a vendor.
    assert len(behaviours["radient"].calls) == 1
    assert per_call_s < BUDGET_CEILING_S


async def test_the_roster_is_serialized_once_per_roster_not_per_message(
    manager, install_legs, monkeypatch
) -> None:
    """A per-message rebuild of a 24-line roster is exactly what the memo prevents."""
    from local_operator.classification import context as context_module
    from local_operator.classification.service import ClassificationService

    install_legs(radient={"script": [choice_response("recommend_skill", "skill-0")]})
    calls: list[str] = []
    real = context_module.candidate_line

    def counting(candidate, limit=None):  # type: ignore[no-untyped-def]
        calls.append(candidate.name)
        return real(candidate, limit)

    monkeypatch.setattr(context_module, "candidate_line", counting)
    subject = ClassificationService(manager=manager, settings=settings())
    for index in range(5):
        await subject.recommend_resources(request(f"message {index}"))

    # 24 lines once, not 24 lines five times.
    assert len(calls) == len(ROSTER)
