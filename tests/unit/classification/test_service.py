"""The service: gates, cache, in-flight sharing, breaker, timeout, notices.

Every test here runs against stub legs (see the ``install_legs`` fixture), so the
suite is hermetic and each failure mode can be produced on demand — which is the
only way several of these paths are reachable at all.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from local_operator.classification.recommend import RecommendationRequest, render_block
from local_operator.classification.service import (
    CACHE_SIZE,
    CIRCUIT_FAILURE_THRESHOLD,
    DEFAULT_AUTO,
    DEFAULT_CASCADE_ATTEMPTS,
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
    assert DEFAULT_AUTO is True
    assert DEFAULT_TIMEOUT_MS == 1500
    assert CACHE_SIZE == 64
    assert CIRCUIT_FAILURE_THRESHOLD == 3


async def test_the_default_is_on_and_off_means_no_vendor_call(manager, install_legs) -> None:
    """The layer is ON when the operator has said nothing (flipped 2026-09-18).

    The absent-key case is the one the flip changed, so it is asserted here
    against the package's own reader: ``{}`` is what a stock install's
    ``config.yml`` carries, and the vendor must be reachable from it.
    """
    subject, legs = service(manager, install_legs, settings_map={}, radient={})
    assert subject.enabled is True

    off, off_legs = service(
        manager, install_legs, settings_map={"classification": {"auto": False}}, radient={}
    )
    assert off.enabled is False
    recommendation = await off.recommend_resources(request())
    assert recommendation.skipped == "disabled"
    assert recommendation.block == ""
    assert off_legs["radient"].calls == []
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
    # The vendor's own token counts ride the outcome too: they are what the
    # harness's cost line prints, and input is what this layer is billed for.
    assert recommendation.input_tokens == 100
    assert recommendation.output_tokens == 10
    assert recommendation.latency_s > 0.0
    assert legs["radient"].calls[0].state["request"] == "deploy core to qa"


async def test_a_leg_that_reports_no_counts_leaves_them_absent(manager, install_legs) -> None:
    """A 200 whose vendor omitted ``usage`` must not become a fabricated zero.

    The whole point of the cost line is that its figures can be believed, and
    ``tokens=0/0`` beside a real ``cost=`` is not a figure a reader can act on.
    The service therefore carries the vendor's absence through as absence — the
    leg's own default response here is the usage-less shape
    (``DecisionResponse`` with no counts set).
    """
    subject, _ = service(manager, install_legs, radient={})
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped is None
    assert recommendation.input_tokens is None
    assert recommendation.output_tokens is None


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
    # Every leg is stubbed: a real leg left in place resolves this machine's own key and
    # answers with ``auth``, which is not weather — that would silently turn the retry
    # walk off and make this test assert the wrong call count.
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [failure]},
        typesafe={"script": [failure]},
        openrouter={"script": [failure]},
    )

    for index in range(CIRCUIT_FAILURE_THRESHOLD):
        recommendation = await subject.recommend_resources(request(f"message {index}"))
        assert recommendation.skipped == "error", index
    # SECOND WALK INCLUDED: one message against a vendor that fails on a retryable
    # kind now spends two attempts (the retry pass) before it is declared failed, and
    # the breaker still counts MESSAGES — three consecutive failures, not six calls.
    assert len(legs["radient"].calls) == CIRCUIT_FAILURE_THRESHOLD * DEFAULT_CASCADE_ATTEMPTS

    with caplog.at_level(logging.WARNING, logger="local_operator.classification.service"):
        opened = await subject.recommend_resources(request("message after the failures"))
    assert opened.skipped == "circuit-open"
    # Open means no call at all, on any later message in this session.
    assert len(legs["radient"].calls) == CIRCUIT_FAILURE_THRESHOLD * DEFAULT_CASCADE_ATTEMPTS
    assert any("circuit breaker opened" in record.message for record in caplog.records)


async def test_a_success_resets_the_consecutive_failure_count(manager, install_legs) -> None:
    failure = DecisionVendorError("down", kind="transport")
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                # Four failures cover the two failing messages that come before the
                # success: each of them spends its retry walk, so each costs two
                # attempts. The trailing entry repeats for everything after.
                failure,
                failure,
                failure,
                failure,
                choice_response("recommend_skill", "minerva-deploy"),
                failure,
            ]
        },
        typesafe={"script": [failure]},
        openrouter={"script": [failure]},
    )
    for index in range(2):
        assert (await subject.recommend_resources(request(f"f{index}"))).skipped == "error"
    assert (await subject.recommend_resources(request("ok"))).skipped is None
    for index in range(2):
        # Two more failures after a success must not open the breaker: the count
        # is CONSECUTIVE, and a stale count would open it on a healthy session.
        assert (await subject.recommend_resources(request(f"g{index}"))).skipped == "error"
    assert (await subject.recommend_resources(request("still trying"))).skipped == "error"
    assert len(legs["radient"].calls) == 11, [call for call in legs["radient"].calls]


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
    # A hit costs nothing: no second call, no reported spend, no wall time — and
    # no tokens either, so the cost line cannot report a spend for a call that
    # was not made.
    assert len(legs["radient"].calls) == 1
    assert second.cost_usd is None
    assert second.input_tokens is None
    assert second.output_tokens is None
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


async def test_a_roster_change_past_the_cap_is_a_cache_hit(manager, install_legs) -> None:
    """A roster change the request cannot carry must not cost a second call.

    ``maxCandidates`` caps the roster per kind before it is sent, so a 13th skill
    on disk changes nothing about the question the vendor is asked. The cache key
    used to be taken over the DISCOVERED list, so that entry moved the key while
    the request body stayed byte-identical — and the session paid for a call it had
    already made (and answered) for the same message.

    Both requests below carry the SAME sent roster: ``maxCandidates: 1`` keeps the
    first skill and drops the second, so the body is identical and the second
    message must be a hit.
    """
    subject, legs = service(
        manager, install_legs, settings_map=settings(maxCandidates=1), radient={}
    )
    message = "deploy core to qa"
    capped = ROSTER[:1]
    await subject.recommend_resources(
        RecommendationRequest(user_message=message, context=None, candidates=capped)
    )
    beyond_cap = tuple(capped) + (
        candidate("minerva-router", kind="skill", description="Routes a request to a skill"),
    )
    await subject.recommend_resources(
        RecommendationRequest(user_message=message, context=None, candidates=beyond_cap)
    )
    assert len(legs["radient"].calls) == 1


async def test_a_roster_change_inside_the_cap_is_a_cache_miss(manager, install_legs) -> None:
    """The other half of the same property: what IS sent still moves the key.

    Without this, keying the cache on nothing at all would pass the test above.
    A replacement entry that survives the cap is a different rubric — a different
    question — and must be asked.
    """
    subject, legs = service(
        manager, install_legs, settings_map=settings(maxCandidates=2), radient={}
    )
    message = "deploy core to qa"
    first = (ROSTER[0], ROSTER[1])
    second = (ROSTER[0], candidate("tunnel-2", kind="guide", description="Another tunnel guide"))
    await subject.recommend_resources(
        RecommendationRequest(user_message=message, context=None, candidates=first)
    )
    await subject.recommend_resources(
        RecommendationRequest(user_message=message, context=None, candidates=second)
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
# The notice is GONE
# ---------------------------------------------------------------------------


async def test_the_seam_publishes_no_notice_at_all(manager, install_legs) -> None:
    """Removal, pinned — not a switch that happens to be off.

    The seam used to render a sentence per message ("Suggestion added for this
    message: <urls>") and the harness painted it under the reply. It is deleted
    outright rather than gated: a `notice` key that turns a surface that no longer
    exists on and off would be a lie in the settings page, and a live branch behind
    it is the second way of doing things this package rejects. This asserts the
    absence so a future re-add has to argue for itself.
    """
    subject, _ = service(
        manager,
        install_legs,
        radient={"script": [choice_response("recommend_skill", "minerva-deploy")]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.resources, "the layer still recommends; only the line went"
    assert not hasattr(subject, "notice")
    assert not hasattr(subject, "notice_enabled")


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
    # An EXPLICIT off: the point is that the disabled layer opens no socket, and
    # the default is now on, so naming ``auto: False`` is what keeps this test
    # about the disabled path rather than about the flip.
    subject, _ = service(
        manager, install_legs, settings_map={"classification": {"auto": False}}, radient={}
    )
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


# ---------------------------------------------------------------------------
# Transient failures: one retry per leg, and a warning that says what happened
# (2026-09-18: the live vendor answers in 540-1500 ms against a 50 ms wait, so a
# single refused connection used to end the pass — and three of them opened the
# breaker for the session, which read as "the feature does not work" rather than
# as a network blip the operator could fix).
# ---------------------------------------------------------------------------


async def test_a_transient_failure_gets_one_retry_on_the_same_leg(manager, install_legs) -> None:
    """A blip must not cost the message its recommendation."""
    subject, legs = service(
        manager,
        install_legs,
        radient={
            "script": [
                DecisionVendorError("connection reset", kind="transport"),
                choice_response("recommend_skill", "minerva-deploy"),
            ]
        },
        typesafe={"script": [DecisionVendorError("down", kind="transport")]},
        openrouter={"script": [DecisionVendorError("down", kind="transport")]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped is None
    assert recommendation.vendor == "radient"
    assert len(legs["radient"].calls) == 2, "the second walk is the retry"
    assert len(legs["typesafe"].calls) == 1, "the first walk still crossed every leg"


async def test_a_transient_failure_that_persists_moves_to_the_next_leg(
    manager, install_legs
) -> None:
    """One retry, not a retry loop: the leg is abandoned after the second attempt."""
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [DecisionVendorError("down", kind="transport")]},
        typesafe={"script": [choice_response("recommend_skill", "minerva-deploy")]},
        openrouter={"script": [DecisionVendorError("down", kind="transport")]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.vendor == "typesafe"
    assert len(legs["radient"].calls) == 1, "failover stays fast: no retry before the next leg"
    assert len(legs["typesafe"].calls) == 1
    assert legs["openrouter"].calls == []


async def test_a_dead_credential_is_not_retried_on_its_own_leg(manager, install_legs) -> None:
    """``auth`` is a property of the leg, not of the moment: ask the NEXT leg instead."""
    subject, legs = service(
        manager,
        install_legs,
        radient={"script": [DecisionVendorError("revoked", kind="auth")]},
        typesafe={"script": [choice_response("recommend_skill", "minerva-deploy")]},
        openrouter={"script": [DecisionVendorError("down", kind="transport")]},
    )
    recommendation = await subject.recommend_resources(request())
    assert recommendation.vendor == "typesafe"
    assert len(legs["radient"].calls) == 1


async def test_a_total_failure_warns_once_with_the_attempts_and_the_legs(
    manager, install_legs, caplog: pytest.LogCaptureFixture
) -> None:
    """The operator's case: providers are configured, the call keeps failing.

    One WARNING, naming which legs were tried and how many attempts each got, so the
    log says "which leg, how many times, and how" instead of only that it failed. The
    message repeats at most ``CIRCUIT_FAILURE_THRESHOLD`` times per session because the
    breaker short-circuits the rest.
    """
    subject, _legs = service(
        manager,
        install_legs,
        settings_map=settings(waitMs=50),
        radient={"script": [DecisionVendorError("down", kind="transport")]},
        typesafe={"script": [DecisionVendorError("down", kind="server")]},
        openrouter={"script": [DecisionVendorError("down", kind="transport")]},
    )
    with caplog.at_level(logging.WARNING, logger="local_operator.classification.service"):
        recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped == "error"
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert any("6 attempt(s)" in line for line in warnings), warnings
    assert any(
        "radient:transport x2" in line and "typesafe:server x2" in line for line in warnings
    ), warnings


async def test_a_pass_with_no_vendor_warns_about_nothing(manager, install_legs, caplog) -> None:
    """No provider is configuration, not weather: silence (§4, and the operator asked)."""
    subject, _legs = service(
        manager,
        install_legs,
        radient={"credential": False},
        typesafe={"credential": False},
        openrouter={"credential": False},
    )
    with caplog.at_level(logging.WARNING, logger="local_operator.classification.service"):
        recommendation = await subject.recommend_resources(request())
    assert recommendation.skipped == "no-vendor"
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


async def test_provider_available_is_false_when_no_leg_has_a_credential(
    manager, install_legs
) -> None:
    subject, _legs = service(
        manager,
        install_legs,
        radient={"credential": False},
        typesafe={"credential": False},
        openrouter={"credential": False},
    )
    assert await subject.provider_available() is False
    # And the probe is a resolution, not a call: nothing was asked.
    assert subject._legs == ()


async def test_provider_available_is_true_when_a_leg_can_serve(manager, install_legs) -> None:
    subject, legs = service(manager, install_legs, radient={"credential": True})
    assert await subject.provider_available() is True
    assert legs["radient"].calls == []
