"""The service: gates, cache, in-flight sharing, breaker, timeout, notices.

Every test here runs against stub legs (see the ``install_legs`` fixture), so the
suite is hermetic and each failure mode can be produced on demand — which is the
only way several of these paths are reachable at all.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest

from local_operator.classification.recommend import (
    Recommendation,
    RecommendationRequest,
    render_block,
)
from local_operator.classification.service import (
    CACHE_SIZE,
    CIRCUIT_FAILURE_THRESHOLD,
    DEFAULT_AUTO,
    DEFAULT_NOTICE,
    DEFAULT_TIMEOUT_MS,
    ClassificationService,
)
from local_operator.classification.types import (
    Answer,
    DecisionResponse,
    DecisionSchemaError,
    DecisionVendorError,
)
from tests.unit.classification.support import candidate, choice_response

pytestmark = pytest.mark.asyncio

ROSTER = [
    candidate("minerva-deploy", kind="skill", description="Deploy a Minerva service"),
    candidate("tunnel", kind="guide", description="Phone access over a tunnel"),
]


def settings(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"auto": True}
    base.update(overrides)
    return {"classification": base}


def request(message: str = "deploy core to qa", **kwargs) -> RecommendationRequest:
    return RecommendationRequest(user_message=message, context=None, candidates=ROSTER, **kwargs)


def service(manager, install_legs, settings_map=None, **leg_specs):
    """A service on stub legs, with the behaviours returned for assertions."""
    behaviours = install_legs(**leg_specs)
    effective = settings() if settings_map is None else settings_map
    return ClassificationService(manager=manager, settings=effective), behaviours


# ---------------------------------------------------------------------------
# Gates (§4)
# ---------------------------------------------------------------------------


async def test_the_defaults_are_the_contract_defaults() -> None:
    assert DEFAULT_AUTO is False
    assert DEFAULT_TIMEOUT_MS == 1500
    assert DEFAULT_NOTICE is True
    assert CACHE_SIZE == 64
    assert CIRCUIT_FAILURE_THRESHOLD == 3


async def test_off_by_default_and_off_means_no_vendor_call(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, settings_map={}, radient={})
    assert subject.enabled is False
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped == "disabled"
    assert recommendation.block == ""
    assert legs["radient"].calls == []


@pytest.mark.parametrize("raw", ["false", False, 0, "no"])
async def test_a_hand_edited_off_reads_as_off(manager, install_legs, raw) -> None:
    subject, _ = service(
        manager, install_legs, settings_map={"classification": {"auto": raw}}, radient={}
    )
    assert subject.enabled is False


async def test_an_empty_roster_never_reaches_a_vendor(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    recommendation = await subject.recommend_resources(
        RecommendationRequest(user_message="hi", context=None, candidates=[])
    )
    assert recommendation.skipped == "empty-roster"
    assert legs["radient"].calls == []


async def test_no_usable_leg_reports_no_vendor(manager, install_legs) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"credential": False},
        typesafe={"credential": False},
        openrouter={"credential": False},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped == "no-vendor"
    assert legs["openrouter"].calls == []


# ---------------------------------------------------------------------------
# Success, and what a success carries
# ---------------------------------------------------------------------------


async def test_a_successful_call_carries_the_vendor_the_cost_and_the_block(
    manager, install_legs
) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [choice_response("recommend_skill", "minerva-deploy", confidence=0.9)]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped is None
    assert [item.resource_url for item in recommendation.resources] == ["skill://minerva-deploy"]
    assert recommendation.block == render_block(recommendation.resources)
    assert recommendation.vendor == "radient"
    assert recommendation.cost_usd == pytest.approx(0.00002)
    assert recommendation.latency_s > 0.0
    assert legs["radient"].calls[0].state["request"] == "deploy core to qa"


async def test_one_question_is_asked_per_kind_with_candidates(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    await subject.recommend_resources(request())
    questions = legs["radient"].calls[0].questions
    assert [question.id for question in questions] == ["recommend_skill", "recommend_guide"]


async def test_a_leg_that_declines_everything_is_a_success_not_a_failure(
    manager, install_legs
) -> None:
    subject, _ = service(
        manager,
        install_legs,
        radient={"script": [choice_response("recommend_skill", "none")]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.resources == ()
    assert recommendation.block == ""
    assert recommendation.skipped is None
    assert recommendation.vendor == "radient"


async def test_the_roster_and_the_context_reach_the_state(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    await subject.recommend_resources(
        RecommendationRequest(
            user_message="hi", context="summary line", candidates=ROSTER, max_recommendations=1
        )
    )
    state = legs["radient"].calls[0].state
    assert state["context"] == "summary line"
    assert state["candidates"] == {
        "skills": ["minerva-deploy: Deploy a Minerva service"],
        "guides": ["tunnel: Phone access over a tunnel"],
    }


# ---------------------------------------------------------------------------
# Cascade within one message
# ---------------------------------------------------------------------------


async def test_the_second_leg_answers_when_the_first_is_down(manager, install_legs) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [DecisionVendorError("down", kind="transport")]},
        typesafe={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.vendor == "typesafe"
    assert len(legs["radient"].calls) == 1
    assert len(legs["typesafe"].calls) == 1


async def test_a_schema_error_never_falls_through_to_the_next_leg(manager, install_legs) -> None:
    """§4: a malformed question is OUR bug, not weather — asking leg two repeats it."""
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [DecisionSchemaError("bad shape", shape="recommend_skill")]},
        typesafe={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped == "error"
    assert legs["typesafe"].calls == []


async def test_a_schema_error_is_logged_loudly_and_disables_that_shape(
    manager, install_legs, caplog
) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                DecisionSchemaError("bad shape", shape="recommend_skill"),
                choice_response("recommend_guide", "tunnel"),
            ]
        },
    )
    with caplog.at_level(logging.ERROR, logger="local_operator.classification.service"):
        first = await subject.recommend_resources(request())
    assert first.skipped == "error"
    assert any("recommend_skill" in record.message for record in caplog.records)

    second = await subject.recommend_resources(request("a different message"))
    # The next question set drops the rejected shape and keeps asking the rest —
    # one bad shape disables that shape, not the feature.
    assert [question.id for question in legs["radient"].calls[1].questions] == ["recommend_guide"]
    assert second.vendor == "radient"


async def test_a_schema_error_that_names_no_shape_disables_nothing(
    manager, install_legs, caplog
) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                DecisionSchemaError("bad body", shape=None),
                choice_response("recommend_skill", "minerva-deploy"),
            ]
        },
    )
    with caplog.at_level(logging.ERROR, logger="local_operator.classification.service"):
        assert (await subject.recommend_resources(request())).skipped == "error"
    second = await subject.recommend_resources(request("another message"))
    assert second.vendor == "radient"
    assert len(legs["radient"].calls[1].questions) == 2


async def test_every_shape_disabled_asks_nothing_and_reports_an_error(
    manager, install_legs
) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                DecisionSchemaError("bad", shape="recommend_skill"),
                DecisionSchemaError("bad", shape="recommend_guide"),
            ]
        },
    )
    assert (await subject.recommend_resources(request())).skipped == "error"
    assert (await subject.recommend_resources(request("message two"))).skipped == "error"
    final = await subject.recommend_resources(request("message three"))
    assert final.skipped == "error"
    # Two calls only: the third attempt had no questions to ask.
    assert len(legs["radient"].calls) == 2


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


async def test_a_slow_leg_times_out_and_is_reported_as_such(manager, install_legs) -> None:
    subject, _ = service(
        manager,
        install_legs,
        settings_map=settings(timeoutMs=50),
        radient={"delay_s": 5.0},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped == "timeout"
    assert recommendation.block == ""
    assert recommendation.latency_s < 1.0


async def test_the_timeout_setting_is_read_in_milliseconds(manager, install_legs) -> None:
    subject, _ = service(manager, install_legs, settings_map=settings(timeoutMs=2500), radient={})
    assert subject.timeout_s == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


async def test_three_consecutive_failures_open_the_breaker_for_the_session(
    manager, install_legs, caplog
) -> None:
    failure = DecisionVendorError("down", kind="transport")
    subject, legs = service(manager, install_legs, radient={"script": [failure]})

    for index in range(CIRCUIT_FAILURE_THRESHOLD):
        recommendation = await subject.recommend_resources(request(f"message {index}"))
        assert recommendation.skipped == "error", index
    assert len(legs["radient"].calls) == CIRCUIT_FAILURE_THRESHOLD

    with caplog.at_level(logging.WARNING, logger="local_operator.classification.service"):
        opened = await subject.recommend_resources(request("message after the failures"))
    assert opened.skipped == "circuit-open"
    # Open means no call at all, on any later message in this session.
    assert len(legs["radient"].calls) == CIRCUIT_FAILURE_THRESHOLD
    assert any("circuit breaker opened" in record.message for record in caplog.records)


async def test_a_success_resets_the_consecutive_failure_count(manager, install_legs) -> None:
    failure = DecisionVendorError("down", kind="transport")
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                failure,
                failure,
                choice_response("recommend_skill", "minerva-deploy"),
                failure,
                failure,
            ]
        },
    )
    for index in range(2):
        assert (await subject.recommend_resources(request(f"f{index}"))).skipped == "error"
    assert (await subject.recommend_resources(request("ok"))).skipped is None
    for index in range(2):
        # Two more failures after a success must not open the breaker: the count
        # is CONSECUTIVE, and a stale count would open it on a healthy session.
        assert (await subject.recommend_resources(request(f"g{index}"))).skipped == "error"
    assert (await subject.recommend_resources(request("still trying"))).skipped == "error"
    assert len(legs["radient"].calls) == 6


async def test_a_schema_error_does_not_count_toward_the_breaker(manager, install_legs) -> None:
    """The breaker exists to stop NETWORK calls; our own bug is not a network call.

    Three consecutive failures would open it, and this test produces three
    ``error`` outcomes — two of them schema rejections. The third is reported as
    ``error`` (nothing left to ask) rather than ``circuit-open``, which is the
    observable difference between the two accounts.
    """
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                DecisionSchemaError("bad", shape="recommend_skill"),
                DecisionSchemaError("bad", shape="recommend_guide"),
            ]
        },
    )
    assert (await subject.recommend_resources(request())).skipped == "error"
    assert (await subject.recommend_resources(request("two"))).skipped == "error"
    assert (await subject.recommend_resources(request("three"))).skipped == "error"
    # Two calls: the third had no questions left, and the breaker is still shut.
    assert len(legs["radient"].calls) == 2


# ---------------------------------------------------------------------------
# Cache and concurrency
# ---------------------------------------------------------------------------


async def test_a_cache_hit_returns_the_same_resources_for_free(manager, install_legs) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    first = await subject.recommend_resources(request())
    second = await subject.recommend_resources(request())
    assert second.resources == first.resources
    assert second.block == first.block
    assert second.vendor == first.vendor
    # A hit costs nothing: no second call, no reported spend, no wall time.
    assert len(legs["radient"].calls) == 1
    assert second.cost_usd is None
    assert second.latency_s == 0.0


async def test_a_changed_message_is_a_different_cache_key(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    await subject.recommend_resources(request("one"))
    await subject.recommend_resources(request("two"))
    assert len(legs["radient"].calls) == 2


async def test_a_changed_roster_is_a_different_cache_key(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    await subject.recommend_resources(request())
    await subject.recommend_resources(
        RecommendationRequest(user_message="deploy core to qa", context=None, candidates=ROSTER[:1])
    )
    assert len(legs["radient"].calls) == 2


async def test_a_cached_result_is_capped_for_a_tighter_caller(manager, install_legs) -> None:
    """The cache key deliberately excludes the cap, so the cap is re-applied on the way out."""
    two_picks = DecisionResponse(
        vendor="stub",
        model="stub-model",
        answers={
            "recommend_skill": Answer(
                id="recommend_skill", kind="choice", value="minerva-deploy", confidence=0.9
            ),
            "recommend_guide": Answer(
                id="recommend_guide", kind="choice", value="tunnel", confidence=0.5
            ),
        },
    )
    subject, legs = service(manager, install_legs, radient={"script": [two_picks]})
    full = await subject.recommend_resources(request(max_recommendations=2))
    assert len(full.resources) == 2
    capped = await subject.recommend_resources(request(max_recommendations=1))
    assert capped.resources == full.resources[:1]
    assert capped.block == render_block(capped.resources)
    assert len(legs["radient"].calls) == 1


async def test_concurrent_callers_share_one_in_flight_call(manager, install_legs) -> None:
    subject, legs = service(
        manager,
        install_legs,
        radient={"delay_s": 0.05, "script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    results = await asyncio.gather(*(subject.recommend_resources(request()) for _ in range(5)))
    assert all(item.vendor == "radient" for item in results)
    # One call for five callers: the second through fifth joined the first, which
    # is what keeps a 0.5 req/s key honest when a burst of callers arrives.
    assert len(legs["radient"].calls) == 1


async def test_the_cache_is_bounded(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    for index in range(CACHE_SIZE + 5):
        await subject.recommend_resources(request(f"message {index}"))
    assert len(subject._cache) == CACHE_SIZE
    # The oldest entries were evicted, so the first message is classified again.
    await subject.recommend_resources(request("message 0"))
    assert len(legs["radient"].calls) == CACHE_SIZE + 6


async def test_the_cache_keeps_the_newest_entry(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={})
    await subject.recommend_resources(request("first"))
    for index in range(4):
        await subject.recommend_resources(request(f"other {index}"))
    # Re-requesting a cached key is free even after later entries were added.
    await subject.recommend_resources(request("first"))
    assert len(legs["radient"].calls) == 5


# ---------------------------------------------------------------------------
# Never raises
# ---------------------------------------------------------------------------


async def test_an_unexpected_exception_from_a_leg_never_escapes(manager, install_legs) -> None:
    subject, _ = service(manager, install_legs, radient={"script": [ValueError("surprise")]})
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped == "error"
    assert recommendation.block == ""


async def test_a_cancelled_turn_stays_cancelled(manager, install_legs) -> None:
    """Cancellation is not a failure mode to convert into an empty recommendation.

    The abandoned call is allowed to finish here: it is shielded, so the shared
    work outlives the caller that started it and its result is cached for the
    next message. The delay is short for that reason — this test is about the
    caller's exception, not about the joiner semantics covered above.
    """
    subject, legs = service(manager, install_legs, radient={"delay_s": 0.05})
    task = asyncio.create_task(subject.recommend_resources(request()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)
    assert len(legs["radient"].calls) == 1


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------


async def test_the_notice_names_the_resources_and_leaves_the_money_out(
    manager, install_legs
) -> None:
    """The design round's copy, pinned: resources and attribution, nothing else.

    The vendor, the six-decimal spend and the duration were all on this line and all
    came off (design round 1, D2/D3): ``$0.000020`` bypassed the repo's one money
    formatter, the 18-character tail is what pushed the line onto a second row at 100
    columns, a cache hit printed ``(0.00s)`` for a call that never happened, and
    ``via <vendor>`` named an implementation leg at a user. The spend still reaches
    the operator — at INFO, from the wiring's own cost log.
    """
    subject, _ = service(
        manager,
        install_legs,
        radient={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    recommendation = await subject.recommend_resources(request())
    line = subject.notice(recommendation)
    assert line is not None
    assert "\n" not in line
    assert "skill://minerva-deploy" in line
    # The three things the design round moved OFF the line.
    assert "radient" not in line
    assert "$" not in line
    assert "s)" not in line
    # Under 80 cells with one resource, which is the row budget D1 asked for.
    assert len(line) <= 80, line


async def test_a_late_answer_is_attributed_to_the_message_it_answers(manager, install_legs) -> None:
    """``late`` is the difference between advice about THIS question and the last one.

    On a real vendor a late answer is the ORDINARY case (a ~250 ms answer against a
    50 ms wait), delivered by the next message — so a line that says "for this
    message" about it is wrong by default, not in an edge case (design round 1, D2).
    """
    subject, _ = service(
        manager,
        install_legs,
        radient={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    fresh = await subject.recommend_resources(request())
    line = subject.notice(fresh)
    assert line is not None and "for this message" in line

    late = dataclasses.replace(fresh, late=True)
    late_line = subject.notice(late)
    assert late_line is not None
    assert "for your previous message" in late_line
    assert "for this message" not in late_line


async def test_the_notice_is_silent_when_switched_off(manager, install_legs) -> None:
    subject, _ = service(
        manager,
        install_legs,
        settings_map=settings(notice=False),
        radient={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    assert subject.notice(await subject.recommend_resources(request())) is None


@pytest.mark.parametrize(
    "leg_spec",
    [
        {"script": [choice_response("recommend_skill", "none")]},
        {"script": [DecisionVendorError("down", kind="transport")]},
    ],
    ids=["vendor-declined", "vendor-down"],
)
async def test_the_notice_says_nothing_when_there_is_nothing_to_announce(
    manager, install_legs, leg_spec
) -> None:
    subject, _ = service(manager, install_legs, radient=leg_spec)
    assert subject.notice(await subject.recommend_resources(request())) is None


async def test_the_notice_never_names_a_vendor(manager, install_legs) -> None:
    """Even when a leg DID answer: a vendor id is not a word a user chose.

    The old line printed ``via unknown vendor`` when no leg answered, which is worse
    — user-facing copy naming an implementation leg — and the fix was to take the
    vendor off the sentence entirely rather than to spell its absence (D3).
    """
    subject, _ = service(manager, install_legs, radient={})
    line = subject.notice(Recommendation(resources=(candidate("x"),), vendor=None))
    assert line is not None and "vendor" not in line
    named = subject.notice(Recommendation(resources=(candidate("x"),), vendor="radient"))
    assert named is not None and "radient" not in named


# ---------------------------------------------------------------------------
# vendor_name
# ---------------------------------------------------------------------------


async def test_vendor_name_reports_the_pin_before_any_call(manager, install_legs) -> None:
    subject, _ = service(
        manager,
        install_legs,
        settings_map=settings(vendor="typesafe"),
        typesafe={},
        radient={},
    )
    assert subject.vendor_name == "typesafe"


async def test_vendor_name_reports_the_leg_that_answered(manager, install_legs) -> None:
    subject, _ = service(manager, install_legs, radient={})
    # Before any call the property can only report what it sees without I/O.
    assert subject.vendor_name == "radient"
    await subject.recommend_resources(request())
    assert subject.vendor_name == "radient"


async def test_vendor_name_after_a_fall_through_names_the_leg_that_answered(
    manager, install_legs
) -> None:
    subject, _ = service(
        manager,
        install_legs,
        radient={"script": [DecisionVendorError("down", kind="transport")]},
        typesafe={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    assert subject.vendor_name == "radient"
    await subject.recommend_resources(request())
    assert subject.vendor_name == "typesafe"


async def test_vendor_name_falls_back_to_the_visible_radients_without_a_pin(
    manager, install_legs
) -> None:
    subject, _ = service(manager, install_legs, radient={})
    assert subject.vendor_name == "radient"


# ---------------------------------------------------------------------------
# The latency budget: one client, one credential resolve, no per-message lookups
# ---------------------------------------------------------------------------


async def test_one_keep_alive_client_is_reused_across_messages(manager, install_legs) -> None:
    """A per-message TLS handshake is 50-150 ms and would blow the budget alone."""
    subject, legs = service(manager, install_legs, radient={})
    await subject.recommend_resources(request("one"))
    client = subject._http
    assert client is not None and not client.is_closed
    await subject.recommend_resources(request("two"))
    # Same object, and the leg was handed it too — not a client of its own.
    assert subject._http is client
    assert legs["radient"].clients[0] is client
    assert len(set(id(item) for item in legs["radient"].clients)) == 1
    await subject.aclose()


async def test_no_client_is_opened_for_a_session_that_never_calls_out(
    manager, install_legs
) -> None:
    subject, _ = service(manager, install_legs, settings_map={}, radient={})
    assert (await subject.recommend_resources(request())).skipped == "disabled"
    assert subject._http is None


async def test_aclose_releases_the_client_and_the_service_still_works(
    manager, install_legs
) -> None:
    subject, _ = service(manager, install_legs, radient={})
    await subject.recommend_resources(request("one"))
    client = subject._http
    assert client is not None
    await subject.aclose()
    assert client.is_closed
    assert subject._http is None
    # Usable again afterwards: a teardown path may run before a later message.
    await subject.recommend_resources(request("two"))
    assert subject._http is not None and subject._http is not client
    await subject.aclose()


async def test_the_credential_is_resolved_once_per_session_not_per_message(
    manager, install_legs
) -> None:
    """``AuthStore.get_api_key`` walks a 7-step SQLite cascade; it may not run per message."""
    subject, legs = service(manager, install_legs, radient={})
    for index in range(3):
        await subject.recommend_resources(request(f"message {index}"))
    assert len(legs["radient"].calls) == 3
    assert legs["radient"].credential_calls == 1
