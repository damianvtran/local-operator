"""``ClassificationService.decide`` — the monitor gate's seam (monitor-tool §8.2).

Every test runs against stub legs (the ``install_legs`` fixture), so each guard
is produced on demand: disabled, no credential, breaker, timeout, schema
rejection, cache, and the fail-open posture that makes ``None`` the one answer
the caller must never treat as "suppress".

The fork itself (which classes suppress) is the monitors suite's subject; this
file is about the call and its guards.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from local_operator.classification.service import (
    CIRCUIT_FAILURE_THRESHOLD,
    DECIDE_CACHE_SIZE,
    DEFAULT_CASCADE_ATTEMPTS,
    ClassificationService,
)
from local_operator.classification.types import (
    Answer,
    DecisionResponse,
    DecisionSchemaError,
    DecisionVendorError,
    Question,
)
from tests.unit.classification.support import candidate, choice_response

pytestmark = pytest.mark.asyncio

#: A stand-in for the monitor gate's question. Deliberately NOT built from
#: ``monitors.classify``: this file tests the seam's mechanics, and importing
#: the monitor package here would couple the two suites (the real question's
#: text is pinned where it is defined).
QUESTION = Question(
    id="monitor_materiality",
    kind="choice",
    instructions="Decide whether this change is MATERIAL.",
    criteria={"material": "yes", "non-material-metadata": "maybe", "ignorable": "no"},
)


def settings(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"auto": True}
    base.update(overrides)
    return {"classification": base}


def service(manager, install_legs, settings_map=None, **leg_specs):
    """A service on stub legs, with the behaviours returned for assertions."""
    behaviours = install_legs(**leg_specs)
    effective = settings() if settings_map is None else settings_map
    return ClassificationService(config_dir=manager, settings=effective), behaviours


async def test_the_contract_defaults_hold() -> None:
    assert DECIDE_CACHE_SIZE == 32


async def test_decide_sends_one_question_and_returns_the_answer(manager, install_legs) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [choice_response("monitor_materiality", "material")]},
    )
    answer = await subject.decide(state="+1/-1 changed lines\n- a\n+ b", question=QUESTION)
    assert isinstance(answer, Answer)
    assert answer.value == "material"
    # One call, carrying exactly the state and the one question: the seam's
    # whole request, per §8.1's one-call-per-changed-monitor rule.
    assert len(legs["radient"].calls) == 1
    request = legs["radient"].calls[0]
    assert request.state == "+1/-1 changed lines\n- a\n+ b"
    assert request.questions == (QUESTION,)


async def test_decide_is_none_when_the_layer_is_off_and_places_no_call(
    manager, install_legs
) -> None:
    subject, legs = service(
        manager, install_legs, settings_map={"classification": {"auto": False}}, radient={}
    )
    assert await subject.decide(state="delta", question=QUESTION) is None
    assert legs["radient"].calls == []


async def test_decide_is_none_without_a_usable_leg_and_places_no_call(
    manager, install_legs
) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"credential": False},
        typesafe={"credential": False},
        openrouter={"credential": False},
    )
    assert await subject.decide(state="delta", question=QUESTION) is None
    assert legs["radient"].calls == [] and legs["typesafe"].calls == []


async def test_decide_falls_through_to_the_next_leg_like_the_message_path(
    manager, install_legs
) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [DecisionVendorError("down", kind="transport")]},
        typesafe={"script": [choice_response("monitor_materiality", "ignorable")]},
    )
    answer = await subject.decide(state="delta", question=QUESTION)
    assert answer is not None and answer.value == "ignorable"
    assert legs["typesafe"].calls, "the second leg must have answered"


async def test_decide_times_out_to_none_on_the_same_deadline(manager, install_legs) -> None:
    subject, _legs = service(
        manager,
        install_legs,
        settings_map=settings(timeoutMs=50),
        radient={"delay_s": 5.0},
    )
    assert await subject.decide(state="delta", question=QUESTION) is None


async def test_decide_failures_count_toward_the_shared_breaker(manager, install_legs) -> None:
    """The breaker is the instance's, so the monitor path and the message path share it."""
    failure = DecisionVendorError("down", kind="transport")
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [failure]},
        typesafe={"script": [failure]},
        openrouter={"script": [failure]},
    )
    for _ in range(CIRCUIT_FAILURE_THRESHOLD):
        assert await subject.decide(state="delta", question=QUESTION) is None
    attempts = len(legs["radient"].calls)
    assert attempts == CIRCUIT_FAILURE_THRESHOLD * DEFAULT_CASCADE_ATTEMPTS  # the retry walk

    # Open: the next decide calls nothing...
    assert await subject.decide(state="another delta", question=QUESTION) is None
    assert len(legs["radient"].calls) == attempts
    # ...and the MESSAGE path is short-circuited by the same breaker: one
    # instance, one breaker (§8.2's "same guards" is not a copy). A non-empty
    # roster so the gate that fires is the breaker, not "empty-roster".
    from local_operator.classification.recommend import RecommendationRequest

    recommendation = await subject.recommend_resources(
        RecommendationRequest(user_message="hi", context=None, candidates=[candidate("x")])
    )
    assert recommendation.skipped == "circuit-open"


async def test_a_success_resets_the_failure_count(manager, install_legs) -> None:
    failure = DecisionVendorError("down", kind="transport")
    subject, _legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                failure,
                failure,
                failure,
                failure,
                choice_response("monitor_materiality", "material"),
            ]
        },
        typesafe={"script": [failure]},
        openrouter={"script": [failure]},
    )
    # Two failing calls (each one walk), then a success: the counter resets, so
    # a later failure is the FIRST consecutive one and the breaker stays shut.
    assert await subject.decide(state="d1", question=QUESTION) is None
    assert await subject.decide(state="d2", question=QUESTION) is None
    assert await subject.decide(state="d3", question=QUESTION) is not None
    assert subject._consecutive_failures == 0


async def test_a_schema_rejection_disables_the_shape_and_never_falls_through(
    manager, install_legs, caplog: pytest.LogCaptureFixture
) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [DecisionSchemaError("bad criteria", shape="monitor_materiality")]},
        typesafe={"script": [choice_response("monitor_materiality", "material")]},
    )
    with caplog.at_level(logging.ERROR, logger="local_operator.classification.service"):
        assert await subject.decide(state="delta", question=QUESTION) is None
    # Our bug, not weather: the next leg must NOT be asked (§4), and a disabled
    # shape means later calls place no request at all.
    assert legs["typesafe"].calls == []
    assert await subject.decide(state="another delta", question=QUESTION) is None
    assert len(legs["radient"].calls) == 1


async def test_decide_caches_by_state(manager, install_legs) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [choice_response("monitor_materiality", "material")]},
    )
    first = await subject.decide(state="same delta", question=QUESTION)
    second = await subject.decide(state="same delta", question=QUESTION)
    assert first is not None and second is first  # the same frozen Answer, not a rebuilt one
    assert len(legs["radient"].calls) == 1
    assert await subject.decide(state="different delta", question=QUESTION) is not None
    assert len(legs["radient"].calls) == 2


async def test_a_question_that_would_be_served_a_different_answer_is_a_miss(
    manager, install_legs
) -> None:
    subject, legs = service(manager, install_legs, radient={})
    other = Question(id="another_question", kind="choice", instructions="x", criteria={"a": "b"})
    await subject.decide(state="delta", question=QUESTION)
    await subject.decide(state="delta", question=other)
    assert len(legs["radient"].calls) == 2


async def test_the_decide_cache_is_bounded(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    for index in range(DECIDE_CACHE_SIZE + 1):
        await subject.decide(state=f"delta {index}", question=QUESTION)
    assert len(legs["radient"].calls) == DECIDE_CACHE_SIZE + 1
    # The oldest entry was evicted: asking for it again is a miss.
    await subject.decide(state="delta 0", question=QUESTION)
    assert len(legs["radient"].calls) == DECIDE_CACHE_SIZE + 2


async def test_a_failure_is_never_cached(manager, install_legs) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={
            # Two failures: one decide call walks the legs twice while every
            # failure is retryable (the cascade's retry walk), so the FIRST
            # call must consume both scripted failures.
            "script": [
                DecisionVendorError("down", kind="transport"),
                DecisionVendorError("down", kind="transport"),
                choice_response("monitor_materiality", "material"),
            ]
        },
        typesafe={"script": [DecisionVendorError("down", kind="transport")]},
        openrouter={"script": [DecisionVendorError("down", kind="transport")]},
    )
    assert await subject.decide(state="delta", question=QUESTION) is None
    answer = await subject.decide(state="delta", question=QUESTION)
    assert answer is not None and answer.value == "material"


async def test_a_200_that_skips_the_question_fails_open_without_counting(
    manager, install_legs
) -> None:
    """A missing answer is a non-answer, not a vendor failure: the breaker must not move."""
    empty = DecisionResponse(vendor="stub", model="stub-model", answers={})
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [empty, empty, empty, choice_response("monitor_materiality", "material")]
        },
    )
    for _ in range(3):
        assert await subject.decide(state=f"d{_}", question=QUESTION) is None
    # Three unanswered calls and the breaker is still shut: the fourth call
    # reaches the vendor and answers.
    assert (await subject.decide(state="final", question=QUESTION)) is not None
    assert len(legs["radient"].calls) == 4


async def test_an_unexpected_exception_from_a_leg_never_escapes(manager, install_legs) -> None:
    subject, _legs = service(
        manager,
        install_legs,
        radient={"script": [RuntimeError("boom")]},
        typesafe={"script": [RuntimeError("boom")]},
        openrouter={"script": [RuntimeError("boom")]},
    )
    assert await subject.decide(state="delta", question=QUESTION) is None


async def test_a_cancelled_decide_stays_cancelled(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={"delay_s": 0.05})
    task = asyncio.create_task(subject.decide(state="delta", question=QUESTION))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(legs["radient"].calls) == 1


async def test_the_cost_line_is_logged_at_info_on_the_vendors_own_figures(
    manager, install_legs, caplog: pytest.LogCaptureFixture
) -> None:
    subject, _legs = service(
        manager,
        install_legs,
        radient={"script": [choice_response("monitor_materiality", "non-material-metadata")]},
    )
    with caplog.at_level(logging.INFO, logger="local_operator.classification.service"):
        await subject.decide(state="delta", question=QUESTION)
    lines = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert any(
        "classification: decide vendor=radient" in line
        and "tokens=100/10" in line
        and "cost=$0.000020" in line
        and "answer=non-material-metadata" in line
        for line in lines
    ), lines


async def test_a_cache_hit_logs_nothing(
    manager, install_legs, caplog: pytest.LogCaptureFixture
) -> None:
    subject, _legs = service(manager, install_legs, radient={})
    await subject.decide(state="delta", question=QUESTION)
    with caplog.at_level(logging.INFO, logger="local_operator.classification.service"):
        await subject.decide(state="delta", question=QUESTION)
    assert [line for line in caplog.records if line.levelno == logging.INFO] == []
