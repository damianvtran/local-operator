"""Public failover requests through real auth selection and synthetic HTTP wire.

No provider network or quota probes: MockTransport records attempted headers and
actual Anthropic bodies while the real stream parser supplies terminal events.
"""

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import replace

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
)
from local_operator.providers import account_affinity as affinity
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import AnthropicClient
from local_operator.providers.failover import ProviderError, stream_with_failover
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport

pytestmark = pytest.mark.asyncio
NOW = 1_800_000_000_000


@pytest.fixture
def routing(tmp_path, monkeypatch):
    clock = {"wall": NOW, "mono": 1000.0, "real_mono": time.monotonic}
    monkeypatch.setattr(affinity.time, "time", lambda: clock["wall"] / 1000)
    monkeypatch.setattr(affinity.time, "monotonic", lambda: clock["mono"])
    store = AuthStore(db_path=tmp_path / "auth.db")
    rows = [
        store.upsert_credential(
            "anthropic",
            {
                "type": "oauth",
                "access": f"synthetic-token-{i}",
                "refresh": "synthetic-refresh",
                "expires": None,
                "email": f"synthetic-{i}@example.invalid",
            },
        )
        for i in range(2)
    ]
    reports = {
        row.id: UsageReport(
            "anthropic",
            fetched_at=NOW,
            limits=[
                UsageLimit(
                    id="anthropic:7d",
                    label="weekly",
                    window="7 day",
                    shared=True,
                    amount=UsageAmount(used_fraction=0.8 if i == 0 else 0.1),
                    resets_at_ms=NOW + (100_000 if i == 0 else 800_000),
                )
            ],
        )
        for i, row in enumerate(rows)
    }
    monkeypatch.setattr(store, "_cached_usage_report", lambda row, provider: reports[row.id])
    quota_calls = []

    async def forbidden_probe(*args, **kwargs):
        quota_calls.append(True)
        raise AssertionError("ranking must not probe provider quota")

    monkeypatch.setattr("local_operator.providers.usage.fetch_usage", forbidden_probe)
    yield store, rows, reports, clock
    store.close()
    assert quota_calls == []


def request():
    return ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-sonnet-5"),
        system_blocks=["Synthetic stable prefix", "Synthetic volatile tail"],
        messages=[Message.user("Synthetic request")],
        prompt_cache_key="synthetic-session",
    )


def response(stop="end_turn"):
    return "".join(
        "data: " + json.dumps(event) + "\n\n"
        for event in [
            {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0, "delta": {"text": "synthetic-ok"}},
            {"type": "message_delta", "delta": {"stop_reason": stop}},
        ]
    ).encode()


async def send(store, attempts, *, req=None, fail=False, retry=None):
    def transport(wire):
        assert str(wire.url) == "https://api.anthropic.com/v1/messages"
        attempts.append((wire.headers.get("authorization"), json.loads(wire.content)))
        return httpx.Response(
            200,
            content=response("refusal" if fail else "end_turn"),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        client = AnthropicClient(http_client=http)

        async def client_for(spec):
            return client

        return [
            event
            async for event in stream_with_failover(
                req or request(),
                store,
                {"retry": {"maxRetries": 0, **(retry or {})}},
                client_for,
                session_id="synthetic-session",
            )
        ]


@pytest.mark.parametrize(
    "retry",
    [
        {"accountSelectionPolicy": "headroom"},
        {"usageAwareAccountPick": True},
        {"usageAwareAccountPick": True, "accountSelectionPolicy": "headroom"},
    ],
)
async def test_direct_public_settings_resolve_headroom_without_configurator(routing, retry):
    store, rows, reports, clock = routing
    attempts = []
    await send(store, attempts, retry=retry)
    assert attempts[-1][0] == f"Bearer {rows[1].data['access']}"
    assert (
        store._account_affinity_store()
        .get(
            "synthetic-session",
            "anthropic",
            "claude-sonnet-5",
            affinity.request_compatibility(request()),
        )
        .account_id
        == rows[1].id
    )


async def test_tentative_pick_is_coherent_only_inside_one_request(routing):
    store, rows, reports, clock = routing
    req = request()
    boundary = store.begin_account_request("synthetic-session", req)
    first = await store.get_oauth_access(
        "anthropic", "synthetic-session", model_id=req.model.model_id
    )
    assert first.credential_id == rows[0].id
    reports[rows[1].id] = replace(
        reports[rows[1].id],
        limits=[replace(reports[rows[1].id].limits[0], resets_at_ms=NOW + 20_000)],
    )
    again = await store.get_oauth_access(
        "anthropic", "synthetic-session", model_id=req.model.model_id
    )
    assert again.credential_id == first.credential_id
    store.end_account_request("synthetic-session", boundary)
    fresh = await store.get_oauth_access(
        "anthropic", "synthetic-session", model_id=req.model.model_id
    )
    assert fresh.credential_id == rows[1].id
    assert not store._account_affinity_store().path_for("synthetic-session").exists()
    older = store.begin_account_request("synthetic-session", req)
    newer = store.begin_account_request("synthetic-session", req)
    store.end_account_request("synthetic-session", older)
    assert store._account_request_boundaries["synthetic-session"] is newer
    store.end_account_request("synthetic-session", newer)


async def test_public_wire_uses_the_explicit_preflight_pick_without_a_fake_lease(routing):
    store, rows, reports, clock = routing
    req = request()
    boundary = store.begin_account_request("synthetic-session", req)
    await store.get_oauth_access("anthropic", "synthetic-session", model_id=req.model.model_id)
    reports[rows[1].id] = replace(
        reports[rows[1].id],
        limits=[replace(reports[rows[1].id].limits[0], resets_at_ms=NOW + 20_000)],
    )
    attempts = []
    try:
        await send(store, attempts)
        assert attempts[-1][0] == f"Bearer {rows[0].data['access']}"
    finally:
        store.end_account_request("synthetic-session", boundary)


async def test_isolated_helper_borrows_main_pointer_but_cannot_pin_next_main_request(routing):
    store, rows, reports, clock = routing
    store.pin_session_credential("anthropic", "synthetic-session", rows[1].id)
    attempts = []
    await send(store, attempts, req=request().model_copy(update={"isolated": True}))
    assert attempts[-1][0] == f"Bearer {rows[1].data['access']}"
    assert store.session_credential_id("anthropic", "synthetic-session") == rows[1].id
    assert not store._account_affinity_store().path_for("synthetic-session").exists()
    main = await store.get_oauth_access(
        "anthropic", "synthetic-session", model_id=request().model.model_id
    )
    assert main.credential_id == rows[0].id


@pytest.mark.parametrize("finish", ["close", "complete", "raise"])
async def test_real_stream_boundary_cleans_up_completion_exception_and_close(
    routing, monkeypatch, finish
):
    from local_operator.harness.types import StreamTextDelta
    from local_operator.model.configure import SessionStreamFn

    store, rows, reports, clock = routing
    stream_fn = SessionStreamFn(store, None, "synthetic-session")

    async def inner(req, signal, boundary=None):
        assert store._account_request_boundaries["synthetic-session"] is boundary
        yield StreamTextDelta(delta="synthetic")
        if finish == "raise":
            raise RuntimeError("synthetic stream failure")

    monkeypatch.setattr(stream_fn, "_stream_account_request", inner)
    stream = stream_fn(request(), None)
    try:
        first = await anext(stream)
        assert isinstance(first, StreamTextDelta)
        assert first.delta == "synthetic"
        assert isinstance(stream, AsyncGenerator)
        if finish == "close":
            await stream.aclose()
        else:
            with pytest.raises(RuntimeError if finish == "raise" else StopAsyncIteration):
                await anext(stream)
        assert "synthetic-session" not in store._account_request_boundaries
    finally:
        assert isinstance(stream, AsyncGenerator)
        await stream.aclose()
        await stream_fn.close()


async def test_real_configurator_preflight_uses_the_fitted_cache_lineage(routing, monkeypatch):
    import asyncio

    from local_operator.model.configure import SessionStreamFn

    store, rows, reports, clock = routing
    monkeypatch.setattr(
        "local_operator.providers.account_affinity.time.monotonic", clock["real_mono"]
    )
    attempts = []

    def transport(wire):
        assert str(wire.url) == "https://api.anthropic.com/v1/messages"
        attempts.append(wire.headers.get("authorization"))
        return httpx.Response(
            200, content=response(), headers={"Content-Type": "text/event-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        stream_fn = SessionStreamFn(
            store,
            {"retry": {"usageAwareFallback": False, "maxRetries": 0}},
            "synthetic-session",
            http_client=client,
        )
        preflight_ids = []

        async def preflight(model):
            access = await store.get_oauth_access(
                model.provider, "synthetic-session", model_id=model.model_id
            )
            preflight_ids.append(access.credential_id)

        monkeypatch.setattr(stream_fn, "preflight_usage", preflight)

        async def collect():
            return [event async for event in stream_fn(request(), None)]

        try:
            assert (await asyncio.wait_for(collect(), 5))[-1].type == "end"
            reports[rows[1].id] = replace(
                reports[rows[1].id],
                limits=[replace(reports[rows[1].id].limits[0], resets_at_ms=NOW + 20_000)],
            )
            assert (await asyncio.wait_for(collect(), 5))[-1].type == "end"
            assert preflight_ids == [rows[0].id, rows[0].id]
            assert attempts == [f"Bearer {rows[0].data['access']}"] * 2
            assert "synthetic-session" not in store._account_request_boundaries
        finally:
            await stream_fn.close()


async def test_no_session_rotates_eligible_rows_without_session_reset_preference(routing):
    store, rows, reports, clock = routing
    selected = [
        (await store.get_oauth_access("anthropic", model_id=request().model.model_id)).credential_id
        for _ in range(4)
    ]
    assert selected == [rows[0].id, rows[1].id] * 2
    reports[rows[0].id] = replace(
        reports[rows[0].id],
        limits=[replace(reports[rows[0].id].limits[0], amount=UsageAmount(used_fraction=1.0))],
    )
    survivors = [
        (await store.get_oauth_access("anthropic", model_id=request().model.model_id)).credential_id
        for _ in range(2)
    ]
    assert survivors == [rows[1].id] * 2


async def test_default_reset_beats_headroom_then_success_lease_expires(routing):
    store, rows, reports, clock = routing
    attempts = []
    await send(store, attempts)
    assert attempts[-1][0] == "Bearer synthetic-token-0"
    assert attempts[-1][1]["system"][0]["cache_control"]["type"] == "ephemeral"
    path = store._account_affinity_store().path_for("synthetic-session")
    receipt = json.loads(path.read_text())["providers"]["anthropic"]
    assert receipt["account_id"] == rows[0].id
    assert receipt["expires_at_ms"] == NOW + 300_000
    assert receipt["compatibility"] == affinity.request_compatibility(request())
    # New rankings prefer the other account; EVERY policy retains success until
    # expiry, without changing the model/effort route.
    reports[rows[0].id].limits[0].resets_at_ms = NOW + 900_000
    reports[rows[1].id].limits[0].resets_at_ms = NOW + 400_000
    for policy in ("headroom", "fair", "reset-soonest"):
        store.configure_account_selection_policy(policy)
        with affinity.selection_context(request()):
            access = await store.get_oauth_access(
                "anthropic", "synthetic-session", model_id="claude-sonnet-5"
            )
        assert access.credential_id == rows[0].id
    assert path.read_text() == json.dumps(json.loads(path.read_text()), sort_keys=True)
    clock.update(wall=NOW + 300_000, mono=1300)
    await send(store, attempts)
    assert attempts[-1][0] == "Bearer synthetic-token-1"


async def test_resume_original_expiry_and_new_settings_signature(routing, monkeypatch):
    store, rows, reports, clock = routing
    await send(store, [])
    resumed = AuthStore(db_path=store.db_path)
    monkeypatch.setattr(resumed, "_cached_usage_report", lambda row, provider: reports[row.id])
    reports[rows[0].id].limits[0].resets_at_ms = NOW + 900_000
    reports[rows[1].id].limits[0].resets_at_ms = NOW + 400_000
    clock.update(wall=NOW + 100_000, mono=1100)
    try:
        with affinity.selection_context(request()):
            access = await resumed.get_oauth_access(
                "anthropic", "synthetic-session", model_id="claude-sonnet-5"
            )
        assert access is not None and access.credential_id == rows[0].id
        french = request().model_copy(
            update={"system_blocks": ["Synthetic French prefix", "Synthetic volatile tail"]}
        )
        attempts = []
        await send(resumed, attempts, req=french)
        assert attempts[-1][0] == "Bearer synthetic-token-1"
        assert "Synthetic French prefix" in str(attempts[-1][1])
        assert "Synthetic stable prefix" not in str(attempts[-1][1])
    finally:
        resumed.close()


async def test_refusal_empty_and_isolated_do_not_create_receipt(routing):
    store, rows, reports, clock = routing
    path = store._account_affinity_store().path_for("synthetic-session")
    await send(store, [], fail=True)
    assert not path.exists()
    await send(store, [], req=request().model_copy(update={"isolated": True}))
    assert not path.exists()

    class Empty:
        async def stream(
            self, request: ChatRequest, api_key: str | None, oauth_access=None
        ) -> AsyncIterator[StreamEvent]:
            if False:
                yield StreamEndEvent(stop_reason="stop")

    async def client_for(spec):
        return Empty()

    assert [
        event
        async for event in stream_with_failover(
            request(), store, None, client_for, session_id="synthetic-session"
        )
    ] == []
    assert not path.exists()


async def test_failure_after_terminal_does_not_anchor_or_renew(routing):
    from local_operator.harness.types import StreamEndEvent, StreamTextDelta

    store, rows, reports, clock = routing
    path = store._account_affinity_store().path_for("synthetic-session")

    class LateFailure:
        async def stream(
            self, request: ChatRequest, api_key: str | None, oauth_access=None
        ) -> AsyncIterator[StreamEvent]:
            # Use the same real marker computation as a built-in transport.
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200))
            ) as http:
                AnthropicClient(http_client=http)._build_body(request, oauth=True)
            yield StreamTextDelta(delta="synthetic output")
            yield StreamEndEvent(stop_reason="stop")
            raise RuntimeError("synthetic failure after terminal")

    async def client_for(spec):
        return LateFailure()

    with pytest.raises(ProviderError, match="synthetic failure after terminal"):
        _ = [
            event
            async for event in stream_with_failover(
                request(),
                store,
                {"retry": {"maxRetries": 0}},
                client_for,
                session_id="synthetic-session",
            )
        ]
    assert not path.exists()
    await send(store, [])
    original = path.read_bytes()
    clock.update(wall=NOW + 10_000, mono=1010)
    with pytest.raises(ProviderError, match="synthetic failure after terminal"):
        _ = [
            event
            async for event in stream_with_failover(
                request(),
                store,
                {"retry": {"maxRetries": 0}},
                client_for,
                session_id="synthetic-session",
            )
        ]
    assert path.read_bytes() != original
    assert json.loads(path.read_text())["providers"] == {}


async def test_low_valid_anchor_survives_other_session_demotion_and_policy_apply(routing):
    store, rows, reports, clock = routing
    await send(store, [])
    reports[rows[0].id].limits[0].amount = UsageAmount(used_fraction=0.99)
    store.deprioritize_credential("anthropic", rows[0].id)
    for policy in ("reset-soonest", "headroom", "fair"):
        store.configure_account_selection_policy(policy)
        with affinity.selection_context(request()):
            access = await store.get_oauth_access(
                "anthropic", "synthetic-session", model_id="claude-sonnet-5"
            )
        assert access.credential_id == rows[0].id


async def test_all_measured_caps_gate_and_no_paid_tier_activation(routing):
    store, rows, reports, clock = routing
    reports[rows[0].id].limits.append(
        replace(
            reports[rows[0].id].limits[0],
            id="anthropic:5h",
            window="5 hour",
            amount=UsageAmount(used_fraction=1),
            resets_at_ms=NOW + 50_000,
        )
    )
    attempts = []
    await send(store, attempts)
    assert attempts[-1][0] == "Bearer synthetic-token-1"
