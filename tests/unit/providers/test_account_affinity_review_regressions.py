"""Review regressions use real public streams and parsed, synthetic wire events.

No provider/quota socket is permitted. In particular, a successful nested helper
must not replace the parent attempt's actually emitted cache-marker lifetime.
"""

import asyncio
import json
import socket
import time
from types import SimpleNamespace

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.providers.account_affinity import request_compatibility
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import (
    AnthropicClient,
    OpenAICompatClient,
    _anthropic_cache_marker_ttl_ms,
)
from local_operator.providers.failover import ProviderError, stream_with_failover
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def no_network_or_quota(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("review regression must not open a real network socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    probes = []

    async def quota(*args, **kwargs):
        probes.append(True)
        raise AssertionError("account selection must not probe quota")

    monkeypatch.setattr("local_operator.providers.usage.fetch_usage", quota)
    yield
    assert probes == []


def encoded_events(events):
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()


@pytest.mark.parametrize(
    "first,second", [("responses", "chat_completions"), ("chat_completions", "responses")]
)
async def test_actual_api_mode_change_rejects_incompatible_lease_in_both_directions(
    tmp_path, first, second
):
    store = AuthStore(db_path=tmp_path / "auth.db")
    rows = [
        store.upsert_credential(
            "openai", {"type": "api_key", "source": "login", "key": f"synthetic-mode-{i}"}
        )
        for i in range(2)
    ]
    remaining = [0.9, 0.2]

    def report(row, provider):
        now = int(time.time() * 1000)
        return UsageReport(
            provider,
            fetched_at=now,
            limits=[
                UsageLimit(
                    id="shared",
                    label="shared",
                    window="7 day",
                    shared=True,
                    amount=UsageAmount(used_fraction=1 - remaining[rows.index(row)]),
                    resets_at_ms=now + 600_000,
                )
            ],
        )

    store._cached_usage_report = report
    req = ChatRequest(
        model=ModelSpec(
            provider="openai",
            model_id="gpt-5.6",
            supports_responses_api=True,
            supports_prompt_cache=True,
            reasoning_effort="high",
        ),
        messages=[Message.user("synthetic question")],
        prompt_cache_key="synthetic-mode",
    )
    attempts = []

    def transport(wire):
        index = next(
            i
            for i, row in enumerate(rows)
            if wire.headers["authorization"] == "Bearer " + row.data["key"]
        )
        attempts.append((wire.url.path, rows[index].id))
        if wire.url.path == "/v1/responses":
            events = [
                {"type": "response.output_text.delta", "delta": "synthetic"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "synthetic",
                        "output": [],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
            ]
        else:
            assert wire.url.path == "/v1/chat/completions"
            events = [
                {"choices": [{"delta": {"content": "synthetic"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        return httpx.Response(
            200, content=encoded_events(events), headers={"Content-Type": "text/event-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:

        async def send(mode):
            client = OpenAICompatClient(
                "https://api.openai.com/v1", http_client=http, openai_api=mode
            )
            return [
                event
                async for event in stream_with_failover(
                    req,
                    store,
                    {
                        "retry": {
                            "maxRetries": 0,
                            "modelFallback": False,
                            "accountSelectionPolicy": "headroom",
                        }
                    },
                    lambda spec: client,
                    session_id="synthetic-mode",
                )
            ]

        try:
            assert (await send(first))[-1].type == "end"
            receipt = store._account_affinity_store().get(
                "synthetic-mode", "openai", req.model.model_id, request_compatibility(req)
            )
            assert receipt.account_id == rows[0].id
            remaining[:] = [0.2, 0.9]
            assert (await send(second))[-1].type == "end"
            assert [account for path, account in attempts] == [rows[0].id, rows[1].id]
            expected_path = "/v1/responses" if second == "responses" else "/v1/chat/completions"
            assert attempts[-1] == (expected_path, rows[1].id)
            store._account_affinity_store().invalidate("synthetic-mode", "openai")
            await send(second)
            assert attempts[-1] == (expected_path, rows[1].id)  # no-anchor control
            assert req.model.reasoning_effort == "high"
            assert req.model.provider == "openai" and req.model.model_id == "gpt-5.6"
        finally:
            store.close()


@pytest.mark.parametrize("parent_tokens,parent_ttl", [(999, 300_000), (1001, 3_600_000)])
@pytest.mark.parametrize("child_outcome", ["success", "failure", "close"])
@pytest.mark.parametrize("different_prefix", [False, True])
async def test_nested_isolated_stream_cannot_replace_parent_wire_facts(
    tmp_path, parent_tokens, parent_ttl, child_outcome, different_prefix
):
    store = AuthStore(db_path=tmp_path / "auth.db")
    store.upsert_credential(
        "anthropic", {"type": "api_key", "source": "login", "key": "synthetic-nested"}
    )
    req = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-sonnet-5"),
        system_blocks=["synthetic prefix", "synthetic tail"],
        messages=[Message.user("synthetic question")],
        prompt_cache_key="synthetic-nested",
        context_tokens_hint=parent_tokens,
    )
    child_tokens = 1001 if parent_tokens == 999 else 999
    child = req.model_copy(
        update={
            "isolated": True,
            "context_tokens_hint": child_tokens,
            **(
                {"system_blocks": ["different child prefix", "synthetic tail"]}
                if different_prefix
                else {}
            ),
        }
    )
    wire_ttls = []

    def transport(wire):
        assert str(wire.url) == "https://api.anthropic.com/v1/messages"
        wire_ttls.append(_anthropic_cache_marker_ttl_ms(json.loads(wire.content)))
        if len(wire_ttls) == 2 and child_outcome == "failure":
            return httpx.Response(
                503,
                json={"error": {"type": "overloaded_error", "message": "synthetic child failure"}},
            )
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0, "delta": {"text": "synthetic"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ]
        return httpx.Response(
            200, content=encoded_events(events), headers={"Content-Type": "text/event-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        client = AnthropicClient(http_client=http, cache_ttl_1h_min_context_tokens=1000)

        def stream(request):
            return stream_with_failover(
                request,
                store,
                {"retry": {"maxRetries": 0, "modelFallback": False}},
                lambda spec: client,
                session_id="synthetic-nested",
            )

        try:
            nested = False
            async for event in stream(req):
                if not nested:
                    nested = True
                    if child_outcome == "failure":
                        with pytest.raises(ProviderError):
                            _ = [part async for part in stream(child)]
                    elif child_outcome == "close":
                        pending = stream(child)
                        await anext(pending)
                        await pending.aclose()
                        await asyncio.sleep(0)
                    else:
                        _ = [part async for part in stream(child)]
            assert wire_ttls == [parent_ttl, 3_600_000 if parent_ttl == 300_000 else 300_000]
            receipt = store._account_affinity_store().get(
                "synthetic-nested", "anthropic", req.model.model_id, request_compatibility(req)
            )
            assert receipt is not None
            assert receipt.expires_at_ms - receipt.sent_at_ms == parent_ttl
            assert receipt.compatibility == request_compatibility(req)
            assert receipt.endpoint_mode == "anthropic:messages"
        finally:
            store.close()


async def test_capture_is_not_ambient_across_yield_and_closes_sync_optional_hook():
    from local_operator.harness.types import StreamTextDelta
    from local_operator.providers.account_affinity import (
        WireHintCapture,
        capture_wire_stream,
        clear_wire_hint,
        set_wire_hint,
        wire_hint,
    )

    req = ChatRequest(model=ModelSpec(provider="anthropic", model_id="claude-test"))
    clear_wire_hint()

    class Source:
        emitted = False
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.emitted:
                raise StopAsyncIteration
            self.emitted = True
            set_wire_hint(req, "anthropic:messages", 300_000, "explicit-wire")
            return StreamTextDelta(delta="synthetic")

        def aclose(self):
            self.closed = True

    source = Source()
    capture = WireHintCapture()
    stream = capture_wire_stream(source, capture)
    assert (await anext(stream)).delta == "synthetic"
    assert wire_hint() is None  # no provider capture remains bound in consumer
    set_wire_hint(req, "anthropic:messages", 3_600_000, "explicit-wire")
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert source.closed and capture.hint.ttl_ms == 300_000
    clear_wire_hint()


async def test_external_session_preflight_then_real_request_uses_current_model_eligibility(
    tmp_path, monkeypatch
):
    from local_operator.model.configure import SessionStreamFn
    from local_operator.session.session import Session

    store = AuthStore(db_path=tmp_path / "auth.db")
    rows = [
        store.upsert_credential(
            "anthropic",
            {
                "type": "oauth",
                "access": f"synthetic-external-{index}",
                "refresh": f"synthetic-refresh-{index}",
                "expires": 10**18,
                "account_id": f"synthetic-account-{index}",
            },
        )
        for index in range(2)
    ]
    now = int(time.time() * 1000)

    def report(index, fable_used):
        return UsageReport(
            "anthropic",
            fetched_at=now,
            limits=[
                UsageLimit(
                    id="shared-five",
                    label="5 hour",
                    window="5h",
                    shared=True,
                    amount=UsageAmount(used=20, limit=100, unit="percent"),
                    resets_at_ms=now + (200_000 if index == 0 else 800_000),
                ),
                UsageLimit(
                    id="fable-week",
                    label="7 day (Fable)",
                    window="7d",
                    shared=False,
                    tier="fable",
                    amount=UsageAmount(used=fable_used, limit=100, unit="percent"),
                    resets_at_ms=now + 600_000,
                ),
            ],
        )

    # Deliberately conflicting reset ranking prefers A. The external diagnosis
    # discovers A's Fable cap is exhausted; its naked pointer is NOT a lease.
    ranking = {row.id: report(index, 20) for index, row in enumerate(rows)}
    monkeypatch.setattr(store, "_cached_usage_report", lambda row, provider: ranking[row.id])
    probes = []

    async def quota(client, provider, *, access_token="", **kwargs):
        index = next(index for index, row in enumerate(rows) if row.data["access"] == access_token)
        probes.append(rows[index].id)
        return report(index, 100 if index == 0 else 20)

    monkeypatch.setattr("local_operator.providers.usage.fetch_usage", quota)
    attempts = []

    def transport(wire):
        index = next(
            index
            for index, row in enumerate(rows)
            if wire.headers["authorization"] == "Bearer " + row.data["access"]
        )
        attempts.append((json.loads(wire.content)["model"], rows[index].id))
        events = [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "synthetic"},
            },
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ]
        return httpx.Response(
            200, content=encoded_events(events), headers={"Content-Type": "text/event-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        stream = SessionStreamFn(
            store,
            {"retry": {"usageAwareFallback": True, "modelFallback": False, "maxRetries": 0}},
            "synthetic-external",
            http_client=http,
        )

        async def refresh_metadata():
            pass

        # Exercise the genuine Session public wrapper without constructing TUI,
        # tools or live subscriptions; only context-display refresh is a stub.
        receiver = SimpleNamespace(
            _stream_fn=stream,
            _model=ModelSpec(provider="anthropic", model_id="claude-fable-5"),
            _refresh_context_metadata=refresh_metadata,
        )
        try:
            await Session.preflight_usage(receiver)
            assert set(probes) == {row.id for row in rows}
            assert not store._account_affinity_store().path_for("synthetic-external").exists()
            probe_count = len(probes)
            request = ChatRequest(
                model=receiver._model,
                system_blocks=["synthetic external prefix"],
                messages=[Message.user("synthetic question")],
            )
            assert [event async for event in stream(request, None)][-1].type == "end"
            assert attempts[-1] == ("claude-fable-5", rows[1].id)
            assert len(probes) == probe_count  # cached knowledge, no new probe for ranking
            assert stream._route_state.active is None
            # The same row's family-only block cannot exclude it from Opus;
            # the new model/policy wins over the historical startup pointer B.
            receiver._model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
            await Session.preflight_usage(receiver, consume_boundary=False)
            request = request.model_copy(update={"model": receiver._model})
            assert [event async for event in stream(request, None)][-1].type == "end"
            assert attempts[-1] == ("claude-opus-5", rows[0].id)
            assert len(probes) == probe_count
            assert "synthetic-external" not in store._account_request_boundaries
        finally:
            await stream.close()
            store.close()
