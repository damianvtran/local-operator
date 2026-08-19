"""Tiny local auto-effort classifier + SessionStreamFn boundary freezing."""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec, StreamEndEvent
from local_operator.model.effort_classifier import (
    PromptEffortClassifier,
    auto_effort_for,
    map_tier_to_effort,
)


def test_classifier_low_for_short_operational_prompt() -> None:
    result = PromptEffortClassifier().classify("show status")
    assert result.tier == "lo"


def test_classifier_medium_for_normal_multi_step_request() -> None:
    result = PromptEffortClassifier().classify(
        "Compare the two config files and explain which setting wins, then list the differences."
    )
    assert result.tier == "med"


def test_classifier_high_for_code_and_acceptance_criteria() -> None:
    result = PromptEffortClassifier().classify("""Implement and review the parser migration.

- Reproduce error 429 in the stream
- Refactor three modules and preserve compatibility
- Add tests for auth, quota, and provider fallback
- Run the release build, then deploy

```python
def parse(value):
    ...
```
""")
    assert result.tier == "hi"


@pytest.mark.parametrize(
    ("tier", "expected"),
    [("lo", "low"), ("med", "medium"), ("hi", "high")],
)
def test_maps_coarse_tiers_without_selecting_minimal_or_max(tier, expected) -> None:
    efforts = ("minimal", "low", "medium", "high", "max")
    assert map_tier_to_effort(tier, efforts) == expected


def test_allow_max_and_models_without_effort() -> None:
    efforts = ("minimal", "low", "medium", "high", "max")
    assert map_tier_to_effort("hi", efforts, allow_max=True) == "max"
    assert map_tier_to_effort("hi", ()) is None


def test_disabled_by_default() -> None:
    assert auto_effort_for("implement x", ("low", "high"), {}) == (None, None)


@pytest.mark.asyncio
async def test_session_stream_fn_freezes_effort_for_one_tool_loop(monkeypatch) -> None:
    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers import failover

    captured: list[ChatRequest] = []

    async def fake_stream(request, *args, **kwargs):
        captured.append(request)
        yield StreamEndEvent(stop_reason="stop")

    monkeypatch.setattr(failover, "stream_with_failover", fake_stream)

    class FakeAuth:
        async def get_oauth_access(self, *args, **kwargs):
            return None

    stream = SessionStreamFn(cast(Any, FakeAuth()), {"effort": {"auto": True}}, "session-x")
    # Avoid usage API work; it still consumes the boundary exactly like prod.
    monkeypatch.setattr(stream, "preflight_usage", lambda model: _noop())
    model = ModelSpec(
        provider="test",
        model_id="m",
        reasoning_efforts=("minimal", "low", "medium", "high", "max"),
        reasoning_effort="medium",
    )
    stream.begin_message()
    simple = ChatRequest(model=model, messages=[Message.user("show status")])
    _ = [event async for event in stream(simple, None)]
    assert captured[-1].model.reasoning_effort == "low"

    # Same user-message tool loop: a large tool result must NOT reclassify.
    follow = ChatRequest(
        model=model,
        messages=[Message.user("show status"), Message.assistant("x" * 5000)],
    )
    _ = [event async for event in stream(follow, None)]
    assert captured[-1].model.reasoning_effort == "low"

    # New message boundary: a complex prompt reclassifies high -> `high`
    # (one rung below max by default).
    stream.begin_message()
    complex_request = ChatRequest(
        model=model,
        messages=[
            Message.user(
                "Implement, refactor, debug, review, deploy, and release this migration.\n"
                + "\n".join(f"- acceptance {i}" for i in range(20))
                + "\n```python\ndef f(): ...\n```"
            )
        ],
    )
    _ = [event async for event in stream(complex_request, None)]
    assert captured[-1].model.reasoning_effort == "high"
    await stream.close()


@pytest.mark.asyncio
async def test_a_mid_message_model_switch_reclassifies_effort(monkeypatch) -> None:
    """A model switched mid-message must not inherit the old model's rung.

    The auto-effort level is frozen for one user message so a tool loop
    reasons at one depth, but it is mapped onto a PARTICULAR model's ladder.
    Carrying it across a switch either sends a rung the new model rejects
    (an HTTP 400 that reads as a broken switch) or silently re-tiers the
    request, so ``on_model_changed`` drops the freeze.
    """
    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers import failover

    captured: list[ChatRequest] = []

    async def fake_stream(request, *args, **kwargs):
        captured.append(request)
        yield StreamEndEvent(stop_reason="stop")

    monkeypatch.setattr(failover, "stream_with_failover", fake_stream)

    class FakeAuth:
        async def get_oauth_access(self, *args, **kwargs):
            return None

    stream = SessionStreamFn(cast(Any, FakeAuth()), {"effort": {"auto": True}}, "session-x")

    async def _preflight(model) -> None:
        # Faithful to prod: the real `preflight_usage` is what CONSUMES the
        # message boundary (it clears the flag before its own early returns),
        # and the freeze under test only exists once the boundary is spent. A
        # stub that skipped this would leave every call reclassifying, and the
        # test would pass whether or not the invalidation works.
        stream._message_boundary_pending = False

    monkeypatch.setattr(stream, "preflight_usage", _preflight)
    wide = ModelSpec(
        provider="test",
        model_id="wide",
        reasoning_efforts=("minimal", "low", "medium", "high", "max"),
    )
    # A DIFFERENT ladder, and deliberately one that does not contain the rung
    # the first classification picks: this is the 400 the invalidation avoids.
    narrow = ModelSpec(provider="test", model_id="narrow", reasoning_efforts=("medium", "high"))

    stream.begin_message()
    _ = [
        event
        async for event in stream(
            ChatRequest(model=wide, messages=[Message.user("show status")]), None
        )
    ]
    assert captured[-1].model.reasoning_effort == "low"

    # The user switches model without sending a new message.
    stream.on_model_changed(narrow)
    _ = [
        event
        async for event in stream(
            ChatRequest(model=narrow, messages=[Message.user("show status")]), None
        )
    ]
    landed = captured[-1].model.reasoning_effort
    assert landed in narrow.reasoning_efforts, landed
    await stream.close()


@pytest.mark.asyncio
async def test_the_preflight_itself_reopens_on_a_switch(monkeypatch) -> None:
    """The quota memo is reopened by `preflight_usage`, not by the hook (review F1).

    The hook used to zero `_usage_checked_at` as well, with a comment claiming
    the old model's 60s TTL would otherwise skip the new model's check. It could
    not: the TTL is gated on the selector matching, and `preflight_usage` clears
    both the route and the timestamp on any selector change — which is the only
    condition under which the hook fires at all. This pins the real mechanism so
    the redundant line is not reintroduced.
    """
    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers.failover import FallbackTarget

    class FakeAuth:
        async def get_oauth_access(self, *args, **kwargs):
            return None

    stream = SessionStreamFn(cast(Any, FakeAuth()), {}, "session-x")
    first = ModelSpec(provider="test", model_id="first")
    await stream.preflight_usage(first)
    assert stream._usage_checked_at != 0.0
    assert stream._primary_selector == "test/first"

    # Switching selector is what reopens the check, with no help from the hook:
    # the timestamp is zeroed and the route dropped the moment the selector
    # differs, BEFORE the TTL comparison is ever reached.
    stream._route_state.active = FallbackTarget("test/fallback", None)
    await stream.preflight_usage(ModelSpec(provider="test", model_id="second"))
    assert stream._primary_selector == "test/second"
    assert stream._route_state.active is None
    await stream.close()


@pytest.mark.asyncio
async def test_a_switch_after_steering_keeps_the_prompts_own_depth(monkeypatch) -> None:
    """A mid-turn switch must not re-read the conversation to pick effort (review F2).

    The classifier reads the newest ``role="user"`` message, and mid-turn that
    is very often NOT the user's prompt: steering, wake, hub, job-result and
    todo-reminder asides all render as user turns. Re-classifying on a switch
    therefore graded the aside — a task opened at ``high`` silently continued at
    ``low`` after a "hurry up" nudge, with an ``auto effort:`` notice for text
    the user never submitted. The remembered TIER is re-mapped instead.
    """
    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers import failover

    captured: list[ChatRequest] = []

    async def fake_stream(request, *args, **kwargs):
        captured.append(request)
        yield StreamEndEvent(stop_reason="stop")

    monkeypatch.setattr(failover, "stream_with_failover", fake_stream)

    class FakeAuth:
        async def get_oauth_access(self, *args, **kwargs):
            return None

    stream = SessionStreamFn(cast(Any, FakeAuth()), {"effort": {"auto": True}}, "session-x")

    async def _preflight(model) -> None:
        stream._message_boundary_pending = False

    monkeypatch.setattr(stream, "preflight_usage", _preflight)
    ladder = ("minimal", "low", "medium", "high", "max")
    first = ModelSpec(provider="test", model_id="first", reasoning_efforts=ladder)
    second = ModelSpec(provider="test", model_id="second", reasoning_efforts=ladder)

    heavy = (
        "Implement, refactor, debug, review, deploy, and release this migration.\n"
        + "\n".join(f"- acceptance {i}" for i in range(20))
        + "\n```python\ndef f(): ...\n```"
    )
    _ = [
        event
        async for event in stream(ChatRequest(model=first, messages=[Message.user(heavy)]), None)
    ]
    opened_at = captured[-1].model.reasoning_effort
    assert opened_at == "high", opened_at

    # Mid-turn: a steering nudge has landed (it renders as a user turn), and the
    # user switches model.
    stream.on_model_changed(second)
    mid_turn = [
        Message.user(heavy),
        Message.assistant("working on it"),
        Message.user("hurry up"),
    ]
    _ = [event async for event in stream(ChatRequest(model=second, messages=mid_turn), None)]
    assert captured[-1].model.reasoning_effort == "high", captured[-1].model.reasoning_effort
    await stream.close()


@pytest.mark.asyncio
async def test_a_switch_with_auto_effort_off_stays_off(monkeypatch) -> None:
    """No classification in force means nothing to re-fit, and nothing invented."""
    from local_operator.model.configure import SessionStreamFn

    class FakeAuth:
        async def get_oauth_access(self, *args, **kwargs):
            return None

    stream = SessionStreamFn(cast(Any, FakeAuth()), {}, "session-x")
    stream.on_model_changed(
        ModelSpec(provider="test", model_id="other", reasoning_efforts=("low",))
    )
    assert stream._message_effort is None
    await stream.close()


async def _noop() -> None:
    return None
