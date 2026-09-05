"""Regression guards for stale admission hints, without live model requests."""

import threading
from unittest.mock import MagicMock

import pytest

from local_operator.compaction.tokens import invalidate_message_cache
from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamUsageEvent,
    TextContent,
    Usage,
)
from local_operator.model.configure import SessionStreamFn
from local_operator.providers.clients import (
    _effective_max_tokens,
    _estimated_prompt_tokens,
)
from local_operator.providers.context import (
    ContextBinding,
    ContextTokenTracker,
    bind_native_context,
    measure_request,
    model_key,
)
from local_operator.providers.failover import ProviderError
from local_operator.providers.replay import native_payload


def request():
    return ChatRequest(
        model=ModelSpec(provider="openai", model_id="gpt-5", context_window=1_000_000),
        messages=[Message.user("starting prompt")],
        system_blocks=["fixed instructions"],
    )


def test_counted_boundary_adds_new_content_and_invalidates_edited_prefix():
    initial = request()
    tracker = ContextTokenTracker()
    tracker.record(measure_request(initial), Usage(context_tokens=100_000))
    assert tracker.estimate(measure_request(initial), 1.1) == 100_000
    grown = initial.model_copy(
        update={"messages": [*initial.messages, Message.user("new result " * 10_000)]}
    )
    estimate = tracker.estimate(measure_request(grown), 1.1)
    assert estimate is not None and estimate > 110_000
    # A compaction/edit preserves neither the counted prefix nor its provider
    # calibration, even if it retains a message's stable transcript id.
    assert isinstance(initial.messages[0].content[0], TextContent)
    initial.messages[0].content[0].text = "edited old history"
    invalidate_message_cache(initial.messages[0])
    assert tracker.estimate(measure_request(grown), 1.1) is None


@pytest.mark.parametrize("changed", ["model", "system", "tools", "compaction", "endpoint"])
def test_context_calibration_cannot_cross_prompt_transformations(changed):
    initial = request()
    tracker = ContextTokenTracker()
    tracker.record(measure_request(initial), Usage(context_tokens=100_000))
    if changed == "model":
        initial.model = initial.model.model_copy(update={"model_id": "other"})
    elif changed == "system":
        initial.system_blocks.append("new task state")
    elif changed == "tools":
        from local_operator.harness.types import AgentTool

        initial.tools.append(AgentTool(name="new", execute=MagicMock()))
    elif changed == "compaction":
        initial.messages = [Message.user("summary")]
    else:
        initial.model = initial.model.model_copy(update={"base_url": "https://other.invalid"})
    assert tracker.estimate(measure_request(initial), 1.1) is None


def test_admission_rejects_unowned_or_other_model_hint():
    initial = request()
    initial.context_tokens_hint = 100_000
    assert _estimated_prompt_tokens(initial)[1] < 100_000
    initial.context_tokens_hint_model = model_key(initial)
    assert _estimated_prompt_tokens(initial) == (100_000, 100_000)
    initial.model = initial.model.model_copy(update={"model_id": "fallback"})
    assert _estimated_prompt_tokens(initial)[1] < 100_000


def test_owned_calibration_above_window_is_not_discarded():
    req = request().model_copy(
        update={
            "model": ModelSpec(
                provider="openai", model_id="gpt-5", context_window=100_000, max_output_tokens=8192
            ),
            "messages": [Message.user("word " * 70_000)],
        }
    )
    tracker = ContextTokenTracker()
    tracker.record(measure_request(req), Usage(context_tokens=90_000))
    req.messages.append(Message.user("more " * 18_000))
    pair = tracker.reconcile(measure_request(req), 1.25)
    assert pair is not None and min(pair) > req.model.context_window
    req.context_tokens_hint, req.context_tokens_hint_measured = pair
    req.context_tokens_hint_model = model_key(req)
    assert _estimated_prompt_tokens(req) == pair
    with pytest.raises(ProviderError, match="prompt is too large"):
        _effective_max_tokens(req)


def test_calibration_margin_cannot_refuse_measured_headroom():
    req = request().model_copy(
        update={
            "model": ModelSpec(
                provider="openai", model_id="gpt-5", context_window=100_000, max_output_tokens=8192
            ),
            "messages": [Message.user("word " * 60_000)],
        }
    )
    tracker = ContextTokenTracker()
    tracker.record(measure_request(req), Usage(context_tokens=60_001))
    req.messages.append(Message.user("more " * 30_000))
    pair = tracker.reconcile(measure_request(req), 1.25)
    assert pair is not None and pair[0] > pair[1]
    req.context_tokens_hint, req.context_tokens_hint_measured = pair
    req.context_tokens_hint_model = model_key(req)
    assert _estimated_prompt_tokens(req) == pair
    assert _effective_max_tokens(req) > 0


def native_answer(req, *, reasoning=5000, scope="account-a", stop="toolUse"):
    return Message.assistant(
        "continue",
        stop_reason=stop,
        usage=Usage(reasoning_tokens=reasoning),
        provider_payload=native_payload(
            req.model,
            "https://api.example/responses",
            "openai-responses",
            [{"type": "reasoning", "encrypted_content": "opaque"}],
            "continue",
            [],
            scope,
        ),
    )


def bind_request(
    req,
    tracker,
    *,
    scope="account-a",
    endpoint="https://api.example/responses",
    protocol="openai-responses",
):
    req.context_binding = ContextBinding(tracker, measure_request(req))
    return bind_native_context(req, endpoint, protocol, scope, 1.25)


def test_native_continuation_counts_reported_reasoning_without_ciphertext_inflation():
    req = request()
    tracker = ContextTokenTracker()
    initial = bind_request(req, tracker)
    tracker.record(initial.context_binding.measured, Usage(context_tokens=40_000))
    answer = native_answer(req)
    req.messages.extend([answer, Message(role="tool", tool_call_id="call", tool_name="read")])
    continued = bind_request(req, tracker)
    assert continued.native_context_tokens == 5000
    assert continued.context_tokens_hint_measured is not None
    assert continued.context_tokens_hint is not None
    assert continued.context_tokens_hint_measured >= 45_000
    # The reported native count is not multiplied by the tokenizer margin.
    assert continued.context_tokens_hint - continued.context_tokens_hint_measured < 100
    before = continued.context_tokens_hint
    assert answer.provider_payload is not None
    answer.provider_payload["native_replay"]["items"][0]["encrypted_content"] *= 10_000
    again = bind_request(req, tracker)
    assert again.context_tokens_hint == before
    assert "context_binding" not in again.model_dump()


@pytest.mark.parametrize("change", ["scope", "endpoint", "protocol", "native", "usage", "user"])
def test_native_calibration_tracks_actual_wire_provenance_and_user_boundaries(change):
    req = request()
    req.messages.append(native_answer(req))
    tracker = ContextTokenTracker()
    initial = bind_request(req, tracker)
    tracker.record(initial.context_binding.measured, Usage(context_tokens=40_000))
    kwargs = {}
    if change in ("scope", "endpoint", "protocol"):
        kwargs[change] = "changed"
    elif change == "native":
        assert req.messages[-1].provider_payload is not None
        req.messages[-1].provider_payload["native_replay"]["items"].append({"type": "reasoning"})
    elif change == "usage":
        assert req.messages[-1].usage is not None
        req.messages[-1].usage.reasoning_tokens += 10
    else:
        req.messages.append(Message.user("a new independent request"))
    result = bind_request(req, tracker, **kwargs)
    assert result.context_tokens_hint_model is None
    if change in ("scope", "endpoint", "protocol", "user"):
        assert result.native_context_tokens == 0


@pytest.mark.parametrize("stop", ["error", "aborted", "length"])
def test_partial_native_outputs_cannot_inflate_admission(stop):
    req = request()
    req.messages.append(native_answer(req, stop=stop))
    result = bind_request(req, ContextTokenTracker())
    assert result.native_context_tokens == 0


def test_cold_resume_counts_valid_native_reasoning_and_google_does_not_count_signature_bytes():
    req = request()
    req.messages.append(native_answer(req))
    req = ChatRequest.model_validate_json(req.model_dump_json())
    result = bind_native_context(
        req, "https://api.example/responses", "openai-responses", "account-a", 1.25
    )
    assert _estimated_prompt_tokens(result)[1] >= 5000
    plain = req.model_copy(update={"messages": [Message.user("task")]})
    text_only = Message.assistant(
        "continue",
        usage=Usage(reasoning_tokens=5000),
        provider_payload=native_payload(
            req.model,
            "https://api.example/responses",
            "openai-responses",
            [{"type": "message"}],
            "continue",
            [],
            "account-a",
        ),
    )
    plain.messages.append(text_only)
    text_only_result = bind_native_context(
        plain, "https://api.example/responses", "openai-responses", "account-a", 1.25
    )
    assert text_only_result.native_context_tokens == 0
    with_native = text_only_result.model_copy(update={"native_context_tokens": 5000})
    scaled, measured = _estimated_prompt_tokens(text_only_result)
    assert _estimated_prompt_tokens(with_native) == (scaled + 5000, measured + 5000)
    google = req.model_copy(update={"model": ModelSpec(provider="google", model_id="gemini-3")})
    google.messages = [
        Message.user("task"),
        Message.assistant(
            "continue",
            usage=Usage(reasoning_tokens=50_000),
            provider_payload=native_payload(
                google.model,
                "google-endpoint",
                "google-content",
                [{"thoughtSignature": "signed"}],
                "continue",
                [],
                "account-a",
            ),
        ),
    ]
    result = bind_native_context(google, "google-endpoint", "google-content", "account-a", 1.25)
    assert result.native_context_tokens == 0


@pytest.mark.asyncio
async def test_warm_tracker_preserves_helper_ttl_suppression(monkeypatch):
    stream = SessionStreamFn(MagicMock(), {}, "ttl-helper")
    initial = request()
    stream._context_tracker.record(measure_request(initial), Usage(context_tokens=100_000))
    seen = []

    async def fake_wire(req, *args, **kwargs):
        seen.append(req)
        yield StreamUsageEvent(usage=Usage(context_tokens=100))

    monkeypatch.setattr("local_operator.providers.failover.stream_with_failover", fake_wire)
    try:
        for hint in (0, 200_000):
            helper = request().model_copy(
                update={"purpose": "compaction", "context_tokens_hint": hint}
            )
            _ = [event async for event in stream(helper, None)]
            assert seen[-1].context_tokens_hint == hint
            assert seen[-1].context_tokens_hint_model is None
        assert stream._context_tracker.provider_tokens == 100_000
    finally:
        await stream.close()


@pytest.mark.asyncio
async def test_stream_reconciles_actual_usage_off_loop_and_separates_child_state(monkeypatch):
    auth = MagicMock()
    parent = SessionStreamFn(auth, {"retry": {"usageAwareFallback": False}}, "parent")
    child = parent.fork("child")
    snapshots = []
    measure_threads = []
    loop_thread = threading.get_ident()
    original_measure = measure_request

    def measure_spy(req):
        measure_threads.append(threading.get_ident())
        return original_measure(req)

    async def preflight(model):
        pass

    async def fake_wire(req, *args, **kwargs):
        snapshots.append((kwargs["session_id"], req))
        yield StreamUsageEvent(usage=Usage(context_tokens=100_000, input_tokens=100_000))

    monkeypatch.setattr("local_operator.providers.context.measure_request", measure_spy)
    monkeypatch.setattr("local_operator.providers.failover.stream_with_failover", fake_wire)
    monkeypatch.setattr(parent, "preflight_usage", preflight)
    monkeypatch.setattr(child, "preflight_usage", preflight)
    parent._message_boundary_pending = False
    parent._message_effort = "high"
    parent._primary_selector = "openai/gpt-5"

    def parent_notice(*args):
        return None

    def child_notice(*args):
        return None

    parent.set_notice_handler(parent_notice)
    child.set_notice_handler(child_notice)
    try:
        # A summary before the turn must neither consume its classification
        # boundary nor replace the effort chosen for user work.
        parent._message_boundary_pending = True
        helper = request().model_copy(update={"purpose": "compaction", "effort_override": "low"})
        _ = [event async for event in parent(helper, None)]
        assert parent._message_boundary_pending
        assert parent._message_effort == "high"
        snapshots.clear()
        parent._message_boundary_pending = False
        initial = request()
        initial.context_tokens_hint = 888_888  # deliberately stale legacy seed
        _ = [event async for event in parent(initial, None)]
        child.begin_message()
        assert parent._message_effort == "high"
        assert parent._primary_selector == "openai/gpt-5"
        assert parent._notice_handler is parent_notice
        assert child._notice_handler is child_notice
        _ = [event async for event in child(request(), None)]
        initial.messages.append(Message.user("new tool output " * 10_000))
        _ = [event async for event in parent(initial, None)]
        assert [owner for owner, _ in snapshots] == ["parent", "child", "parent"]
        assert snapshots[0][1].context_tokens_hint == 888_888
        assert snapshots[0][1].context_tokens_hint_model is None
        assert _estimated_prompt_tokens(snapshots[0][1])[1] < 888_888
        assert snapshots[1][1].context_tokens_hint is None
        assert snapshots[2][1].context_tokens_hint > 100_000
        assert all(tid != loop_thread for tid in measure_threads)
        assert child._http is parent._http
        assert child._route_state is not parent._route_state
        await parent.close()
        assert not child._http.is_closed
        await parent.close()  # idempotent, cannot consume the child's ownership
        assert not child._http.is_closed
    finally:
        await parent.close()
        await child.close()
    assert child._http.is_closed
